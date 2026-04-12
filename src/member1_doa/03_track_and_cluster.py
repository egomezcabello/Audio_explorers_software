#!/usr/bin/env python3
"""
03_track_and_cluster.py – Global-direction tracking with quality filtering.
============================================================================
Fourth step of the Member 1 (DoA) pipeline.

What it does
------------
Takes the angular power map P(θ, t) from step 02 and extracts discrete
speaker candidates with smoothed DoA tracks and active-segment times.

Algorithm
---------
1.  **Find global dominant directions** from the *time-averaged* angular
    power spectrum.

1b. **Windowed discovery** (optional): slide short time windows (~1.5 s)
    across the recording, detect local peaks in each window, and
    cluster repeated detections.  This recovers intermittent or quiet
    speakers that the full-file average misses.

1c. **Merge** global + windowed directions, de-duplicating angles that
    are too close.

2.  **Selective per-frame peak extraction** with a second-peak gate.

3.  **Assign peaks to merged directions**.

3b. **Pre-split broad clusters**: if a cluster's angular spread is too
    wide (bimodal), split it into two sub-clusters using 2-means
    before track building.

4.  **Track building & smoothing**: per-direction circular-mean track,
    Gaussian smoothing, active-segment extraction.

5.  **Quality filtering**: reject tracks that are too few, too short,
    too weak, or too angularly diffuse.

6.  **Example validation** (example tag only).

Outputs
-------
- ``outputs/doa/{tag}_doa_tracks.json``
- ``outputs/doa/doa_tracks.json``  (canonical, for mixture)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.common.config import CFG, get_stft_params
from src.common.constants import SAMPLE_RATE
from src.common.logging_utils import setup_logging
from src.common.paths import DOA_DIR, ensure_output_dirs

logger = setup_logging(__name__)


# ── helpers ────────────────────────────────────────────────────────────

def _angular_distance(a: float, b: float) -> float:
    """Shortest angular distance in degrees (0–180)."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def _circular_mean(angles_deg: np.ndarray) -> float:
    """Circular mean of angles in degrees → value in [0, 360)."""
    rad = np.deg2rad(angles_deg)
    return float(np.degrees(np.arctan2(
        np.mean(np.sin(rad)), np.mean(np.cos(rad))))) % 360.0


def _circular_std(angles_deg: np.ndarray) -> float:
    """Circular standard deviation of angles in degrees."""
    rad = np.deg2rad(angles_deg)
    R = np.sqrt(np.mean(np.cos(rad)) ** 2 + np.mean(np.sin(rad)) ** 2)
    R = max(R, 1e-10)
    return float(np.degrees(np.sqrt(-2.0 * np.log(R))))


# ── 1. Global dominant directions ─────────────────────────────────────

def find_global_directions(
    P: np.ndarray,
    min_distance_deg: int = 40,
    rel_threshold: float = 0.10,
    prominence_frac: float = 0.10,
    max_speakers: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find dominant speaker directions from the time-averaged angular
    power spectrum.

    Parameters
    ----------
    P : np.ndarray, shape (n_grid, n_frames)
    min_distance_deg : int
        Minimum angular separation between returned peaks.
    rel_threshold : float
        Peaks must exceed this fraction of the global maximum.
    prominence_frac : float
        Peaks must have prominence ≥ this fraction of global max.
    max_speakers : int
        Hard cap on the number of directions returned.

    Returns
    -------
    directions : np.ndarray  – peak angles in degrees (sorted)
    heights    : np.ndarray  – corresponding peak heights
    """
    n_grid = P.shape[0]
    avg = P.mean(axis=1)
    avg_max = avg.max()
    if avg_max < 1e-12:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    height_thr = rel_threshold * avg_max
    prom_thr = prominence_frac * avg_max

    # Circular-safe peak detection: tile the spectrum
    pad = min_distance_deg + 2
    tiled = np.concatenate([avg[-pad:], avg, avg[:pad]])
    peaks, props = find_peaks(
        tiled,
        height=height_thr,
        distance=min_distance_deg,
        prominence=prom_thr,
    )
    peaks_orig = peaks - pad
    valid = (peaks_orig >= 0) & (peaks_orig < n_grid)
    peaks_orig = peaks_orig[valid]
    peak_heights = props["peak_heights"][valid]

    if len(peaks_orig) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Keep top-max_speakers by height
    order = np.argsort(peak_heights)[::-1][:max_speakers]
    dirs = peaks_orig[order].astype(np.float64)
    hts = peak_heights[order]
    # Sort by angle for deterministic ordering
    sort_idx = np.argsort(dirs)
    return dirs[sort_idx], hts[sort_idx]


# ── 1b. Windowed speaker-direction discovery ─────────────────────────

def find_windowed_directions(
    P: np.ndarray,
    hop_length: int,
    sr: int,
    window_dur_s: float = 1.5,
    window_overlap: float = 0.5,
    min_distance_deg: int = 40,
    rel_threshold: float = 0.15,
    min_window_count: int = 3,
    max_new_dirs: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discover speaker directions from short sliding windows over time.

    This catches intermittent or quiet speakers that don't dominate
    the global time average.

    The method slides a window across the posterior, picks peaks in
    each window's time-averaged spectrum, accumulates a histogram of
    detected angles, then finds persistent directions that appeared
    in at least ``min_window_count`` windows.

    Parameters
    ----------
    P : (n_grid, n_frames)
    hop_length, sr : for frame-to-second conversion
    window_dur_s : window length in seconds
    window_overlap : overlap fraction (0 to 0.99)
    min_distance_deg : min angular separation between peaks
    rel_threshold : peaks must exceed this fraction of window max
    min_window_count : direction must appear in ≥ this many windows
    max_new_dirs : cap on number of windowed directions returned

    Returns
    -------
    directions : peak angles in degrees (sorted)
    heights : corresponding accumulated heights
    """
    n_grid, n_frames = P.shape
    win_frames = max(1, int(window_dur_s * sr / hop_length))
    step_frames = max(1, int(win_frames * (1.0 - window_overlap)))

    # Accumulate: weighted histogram of detected peak angles
    peak_histogram = np.zeros(n_grid, dtype=np.float64)
    peak_count = np.zeros(n_grid, dtype=np.int32)

    n_windows = 0
    t = 0
    while t + win_frames <= n_frames:
        win_avg = P[:, t:t + win_frames].mean(axis=1)
        win_max = win_avg.max()
        if win_max < 1e-10:
            t += step_frames
            continue

        height_thr = rel_threshold * win_max
        pad = min_distance_deg + 2
        tiled = np.concatenate([win_avg[-pad:], win_avg, win_avg[:pad]])
        pks, props = find_peaks(tiled, height=height_thr,
                                distance=min_distance_deg)
        pks_orig = pks - pad
        vmask = (pks_orig >= 0) & (pks_orig < n_grid)
        pks_orig = pks_orig[vmask]
        pheights = props["peak_heights"][vmask]

        for pk, ht in zip(pks_orig, pheights):
            # Spread over ±2 bins to handle slight angle jitter
            for offset in range(-2, 3):
                idx = (pk + offset) % n_grid
                w = max(0.0, 1.0 - abs(offset) * 0.3)
                peak_histogram[idx] += ht * w
            peak_count[pk] += 1

        n_windows += 1
        t += step_frames

    if n_windows == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Keep only bins with enough window appearances
    persistent = peak_histogram.copy()
    persistent[peak_count < min_window_count] = 0.0

    if persistent.max() < 1e-10:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    pad = min_distance_deg + 2
    tiled = np.concatenate([persistent[-pad:], persistent, persistent[:pad]])
    pks, _ = find_peaks(tiled, distance=min_distance_deg)
    pks_orig = pks - pad
    vmask = (pks_orig >= 0) & (pks_orig < n_grid)
    pks_orig = pks_orig[vmask]
    ht_vals = tiled[pks[vmask]]

    if len(pks_orig) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    order = np.argsort(ht_vals)[::-1][:max_new_dirs]
    dirs = pks_orig[order].astype(np.float64)
    hts = ht_vals[order]
    sort_idx = np.argsort(dirs)
    return dirs[sort_idx], hts[sort_idx]


# ── 1c. Merge direction sets ─────────────────────────────────────────

def _merge_direction_sets(
    global_dirs: np.ndarray,
    global_heights: np.ndarray,
    windowed_dirs: np.ndarray,
    windowed_heights: np.ndarray,
    min_sep_deg: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Merge global and windowed directions, removing windowed duplicates
    that are too close to an existing direction.

    Returns
    -------
    merged_dirs, merged_heights, sources
        ``sources[i]`` is ``"global"`` or ``"windowed"``.
    """
    merged = list(global_dirs)
    heights = list(global_heights)
    sources: List[str] = ["global"] * len(global_dirs)

    for wdir, wht in zip(windowed_dirs, windowed_heights):
        too_close = any(
            _angular_distance(wdir, d) < min_sep_deg for d in merged
        )
        if not too_close:
            merged.append(wdir)
            heights.append(wht)
            sources.append("windowed")

    if not merged:
        return np.array([]), np.array([]), []

    arr_d = np.array(merged, dtype=np.float64)
    arr_h = np.array(heights, dtype=np.float64)
    sort_idx = np.argsort(arr_d)
    return arr_d[sort_idx], arr_h[sort_idx], [sources[i] for i in sort_idx]


# ── 2. Selective per-frame peak extraction ────────────────────────────

def extract_peaks_per_frame(
    P: np.ndarray,
    top_k: int = 3,
    min_distance_deg: int = 40,
    rel_threshold: float = 0.15,
    second_peak_ratio: float = 0.50,
    temporal_persist_window: int = 5,
    temporal_persist_count: int = 3,
    temporal_persist_tol_deg: float = 8.0,
) -> np.ndarray:
    """
    Extract per-frame peaks with a second-peak gate and temporal
    persistence rescue.

    The strongest peak in each frame is always kept (if it exceeds
    ``rel_threshold``).  Additional peaks are kept only if their height
    is ≥ ``second_peak_ratio × strongest_peak_in_frame``.

    **Temporal persistence rescue**: secondary peaks that fail the
    height ratio gate but recur at a similar angle in ≥
    ``temporal_persist_count`` of the surrounding
    ``±temporal_persist_window`` frames are rescued.  This is a more
    reliable way to detect weak nearby speakers than lowering the
    ratio globally (which increases false positives).

    Parameters
    ----------
    P : np.ndarray, shape (n_grid, n_frames)
    top_k : int
        Maximum peaks per frame.
    min_distance_deg : int
        Minimum angular separation between peaks.
    rel_threshold : float
        Absolute floor: peaks < ``rel_threshold × frame_max`` are dropped.
    second_peak_ratio : float
        Secondary peaks must reach this fraction of the frame's
        strongest peak.  0.5 means the second peak must be at least
        half as tall as the first.
    temporal_persist_window : int
        Half-width of the temporal neighbourhood (in frames) checked
        for persistence of secondary peaks.
    temporal_persist_count : int
        Minimum recurrences of a secondary peak within the window
        for it to be rescued.
    temporal_persist_tol_deg : float
        Angular tolerance (in degrees) for matching secondary peaks
        across frames.

    Returns
    -------
    points : np.ndarray, shape (N, 3)
        Each row is ``(frame_idx, azimuth_deg, score)``.
    """
    n_grid, n_frames = P.shape
    all_points: List[Tuple[int, float, float]] = []
    # Track ALL detected peaks per frame (including below gate)
    # for temporal persistence rescue
    all_peaks_per_frame: List[List[Tuple[float, float]]] = []

    for t in range(n_frames):
        spectrum = P[:, t]
        frame_max = spectrum.max()
        if frame_max < 1e-8:
            all_peaks_per_frame.append([])
            continue

        height_thr = rel_threshold * frame_max

        # Circular peak detection via tiling
        pad = min_distance_deg + 2
        tiled = np.concatenate([spectrum[-pad:], spectrum, spectrum[:pad]])
        peaks, props = find_peaks(
            tiled, height=height_thr, distance=min_distance_deg,
        )
        peaks_orig = peaks - pad
        valid_mask = (peaks_orig >= 0) & (peaks_orig < n_grid)
        peaks_orig = peaks_orig[valid_mask]
        heights = props["peak_heights"][valid_mask]

        if len(peaks_orig) == 0:
            all_peaks_per_frame.append([])
            continue

        # Sort by height (strongest first)
        order = np.argsort(heights)[::-1][:top_k]
        best_height = heights[order[0]]

        frame_peaks: List[Tuple[float, float]] = []
        for rank, idx in enumerate(order):
            h = heights[idx]
            az = float(peaks_orig[idx])
            frame_peaks.append((az, float(h)))
            # Always accept the strongest; gate the rest
            if rank > 0 and h < second_peak_ratio * best_height:
                continue
            all_points.append((t, az, float(h)))

        all_peaks_per_frame.append(frame_peaks)

    # ── Temporal persistence rescue ───────────────────────────────────
    # For each frame, find secondary peaks that were below the gate but
    # recur consistently in neighbouring frames.
    if temporal_persist_window > 0 and temporal_persist_count > 1:
        accepted_set: set = set()  # (frame, round(az)) already accepted
        for t, az, _ in all_points:
            accepted_set.add((t, round(az)))

        rescued: List[Tuple[int, float, float]] = []
        for t in range(n_frames):
            frame_peaks = all_peaks_per_frame[t]
            if len(frame_peaks) < 2:
                continue

            # Check non-primary peaks (skip index 0 which is strongest)
            for az, h in frame_peaks[1:]:
                if (t, round(az)) in accepted_set:
                    continue

                # Count how many neighbouring frames have a peak near az
                cnt = 0
                t_lo = max(0, t - temporal_persist_window)
                t_hi = min(n_frames, t + temporal_persist_window + 1)
                for tt in range(t_lo, t_hi):
                    if tt == t:
                        continue
                    for nb_az, _nb_h in all_peaks_per_frame[tt]:
                        if _angular_distance(az, nb_az) <= temporal_persist_tol_deg:
                            cnt += 1
                            break

                if cnt >= temporal_persist_count:
                    rescued.append((t, az, h))
                    accepted_set.add((t, round(az)))

        if rescued:
            logger.info("  temporal persistence rescued %d secondary "
                        "peaks", len(rescued))
            all_points.extend(rescued)

    if not all_points:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(all_points, dtype=np.float64)


# ── 3. Assign peaks to global directions ──────────────────────────────

def assign_peaks_to_directions(
    points: np.ndarray,
    directions: np.ndarray,
    max_distance_deg: float = 25.0,
) -> np.ndarray:
    """
    Assign each per-frame peak to the nearest global direction.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    directions : np.ndarray, shape (K,)
        Global direction angles in degrees.
    max_distance_deg : float
        Points farther than this from every direction are labelled −1.

    Returns
    -------
    labels : np.ndarray, shape (N,)
        Cluster index (0 … K-1) or −1 for noise.
    """
    N = points.shape[0]
    K = len(directions)
    labels = -np.ones(N, dtype=int)

    for i in range(N):
        az = points[i, 1]
        best_dist = max_distance_deg + 1.0
        best_k = -1
        for k in range(K):
            d = _angular_distance(az, directions[k])
            if d < best_dist:
                best_dist = d
                best_k = k
        if best_dist <= max_distance_deg:
            labels[i] = best_k

    return labels


# ── 3b. Pre-split broad clusters ─────────────────────────────────────

def _presplit_broad_clusters(
    points: np.ndarray,
    labels: np.ndarray,
    split_std_deg: float = 18.0,
    split_min_points: int = 30,
    max_temporal_overlap: float = 0.70,
) -> Tuple[np.ndarray, List[str]]:
    """
    Inspect each cluster of assigned peaks.  If a cluster's angular
    spread is wider than ``split_std_deg``, attempt to split it into
    two sub-clusters using 2-means on the unwrapped angle.

    The split is only accepted if:
    1. Both sub-clusters have at least ``split_min_points`` points.
    2. The two centroids are at least 10° apart.
    3. Temporal overlap between the two children is below
       ``max_temporal_overlap`` (prevents splitting one broad-peaked
       speaker into two identities due to DoA jitter).

    Modifies ``labels`` in-place.

    Returns
    -------
    labels : the (modified) label array
    split_log : human-readable description of each split event
    """
    from sklearn.cluster import KMeans

    unique = sorted(set(labels.tolist()) - {-1})
    if not unique:
        return labels, []

    next_label = max(unique) + 1
    split_log: List[str] = []

    for k in unique:
        mask = labels == k
        n_pts = int(mask.sum())
        if n_pts < 2 * split_min_points:
            continue

        azimuths = points[mask, 1]
        circ_std = _circular_std(azimuths)

        if circ_std < split_std_deg:
            continue

        # Try 2-means on angle (unwrapped relative to circular mean)
        rad = np.deg2rad(azimuths)
        mean_rad = np.arctan2(np.mean(np.sin(rad)),
                              np.mean(np.cos(rad)))
        diff = np.arctan2(np.sin(rad - mean_rad),
                          np.cos(rad - mean_rad))

        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        sub = km.fit_predict(diff.reshape(-1, 1))

        n0, n1 = int((sub == 0).sum()), int((sub == 1).sum())
        if n0 < split_min_points or n1 < split_min_points:
            continue

        mean0 = _circular_mean(azimuths[sub == 0])
        mean1 = _circular_mean(azimuths[sub == 1])
        sep = _angular_distance(mean0, mean1)

        if sep < 10.0:  # too close — not a real split
            continue

        # Temporal overlap check: if both children are active in
        # mostly the same frames, it's likely one broad speaker
        # experiencing DoA jitter, not two real sources.
        frames_k = points[mask, 0].astype(int)
        frames_0 = set(frames_k[sub == 0].tolist())
        frames_1 = set(frames_k[sub == 1].tolist())
        intersection = len(frames_0 & frames_1)
        union = len(frames_0 | frames_1)
        t_overlap = intersection / max(union, 1)

        if t_overlap > max_temporal_overlap:
            logger.debug("  presplit reject %.0f°: temporal overlap=%.2f "
                         "(>%.2f)",
                         _circular_mean(azimuths), t_overlap,
                         max_temporal_overlap)
            continue

        # Accept the split: reassign sub-cluster 1 to a new label
        indices = np.where(mask)[0]
        labels[indices[sub == 1]] = next_label
        next_label += 1

        original_az = _circular_mean(azimuths)
        split_log.append(
            f"{original_az:.0f}°→{mean0:.0f}°+{mean1:.0f}° "
            f"(sep={sep:.0f}°, t_iou={t_overlap:.2f})"
        )

    return labels, split_log


# ── 4. Track building & smoothing ─────────────────────────────────────

def build_tracks(
    points: np.ndarray,
    labels: np.ndarray,
    n_frames: int,
    hop_length: int,
    smooth_sigma: int = 5,
    segment_merge_gap: int = 20,
    min_segment_dur_s: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Build one smoothed track per direction.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        ``(frame_idx, azimuth_deg, score)``
    labels : np.ndarray, shape (N,)
    n_frames : int
    hop_length : int
    smooth_sigma : int
    segment_merge_gap : int
        Merge active segments separated by fewer frames than this.
    min_segment_dur_s : float
        Drop segments shorter than this.

    Returns
    -------
    tracks : list[dict]
    """
    sr = SAMPLE_RATE
    unique_labels = sorted(set(labels) - {-1})
    tracks: List[Dict[str, Any]] = []

    for k in unique_labels:
        mask = labels == k
        pk = points[mask]
        frames = pk[:, 0].astype(int)
        azimuths = pk[:, 1]
        scores = pk[:, 2]

        mean_az = _circular_mean(azimuths)
        mean_score = float(scores.mean())

        # Circular std of assigned points (before smoothing)
        angular_std = _circular_std(azimuths) if len(azimuths) >= 2 else 0.0

        # Per-frame circular mean azimuth
        az_per_frame = np.full(n_frames, np.nan)
        for f in np.unique(frames):
            fm = frames == f
            az_per_frame[f] = _circular_mean(azimuths[fm])

        valid = ~np.isnan(az_per_frame)
        if valid.sum() < 2:
            continue

        # Unwrap → interpolate → Gaussian smooth
        vi = np.where(valid)[0]
        vv = np.deg2rad(az_per_frame[vi])
        unwrapped = np.unwrap(vv)
        interp = np.interp(np.arange(n_frames), vi, unwrapped)
        smoothed = np.degrees(gaussian_filter1d(interp, sigma=smooth_sigma)) % 360.0

        # Build active segments (merge small gaps) — keep frame ranges
        active_frames = np.sort(np.unique(frames))
        segments: List[List[float]] = []
        seg_frame_ranges: List[Tuple[int, int]] = []
        if len(active_frames) > 0:
            seg_start = active_frames[0]
            prev = active_frames[0]
            for fi in active_frames[1:]:
                if fi - prev > segment_merge_gap:
                    t0 = round(float(seg_start * hop_length / sr), 3)
                    t1 = round(float(prev * hop_length / sr), 3)
                    if t1 - t0 >= min_segment_dur_s:
                        segments.append([t0, t1])
                        seg_frame_ranges.append((int(seg_start), int(prev)))
                    seg_start = fi
                prev = fi
            t0 = round(float(seg_start * hop_length / sr), 3)
            t1 = round(float(prev * hop_length / sr), 3)
            if t1 - t0 >= min_segment_dur_s:
                segments.append([t0, t1])
                seg_frame_ranges.append((int(seg_start), int(prev)))

        if not segments:
            continue

        total_dur = sum(s[1] - s[0] for s in segments)

        # Azimuth track: fill only within active segments (None in gaps)
        track_az = np.full(n_frames, np.nan)
        for f_start, f_end in seg_frame_ranges:
            track_az[f_start:f_end + 1] = smoothed[f_start:f_end + 1]

        # doa_track: [[frame_idx, azimuth_deg], ...]
        doa_track: List[List[float]] = [
            [f_idx, round(float(track_az[f_idx]), 1)]
            for f_idx in range(n_frames) if not np.isnan(track_az[f_idx])
        ]

        first_f = int(active_frames[0])
        last_f = int(active_frames[-1])

        tracks.append({
            "id": "",                         # assigned after filtering
            "mean_azimuth": round(mean_az, 1),
            "mean_score": round(mean_score, 4),
            "n_points": int(mask.sum()),
            "angular_std_deg": round(angular_std, 1),
            "total_duration_s": round(total_dur, 3),
            "first_active_time_s": round(first_f * hop_length / sr, 3),
            "last_active_time_s": round(last_f * hop_length / sr, 3),
            "n_active_segments": len(segments),
            "doa_track": doa_track,
            "active_segments": segments,
            "azimuth_track": [
                round(float(v), 1) if not np.isnan(v) else None
                for v in track_az
            ],
        })

    return tracks


# ── 4b. Merge close & overlapping tracks ──────────────────────────────

def merge_tracks_if_close_and_overlapping(
    tracks: List[Dict[str, Any]],
    max_sep_deg: float = 12.0,
    min_temporal_iou: float = 0.60,
) -> List[Dict[str, Any]]:
    """
    Merge tracks that are close in angle and heavily overlap in time.

    Two tracks that overlap temporally with IoU > ``min_temporal_iou``
    and whose mean azimuths are within ``max_sep_deg`` almost certainly
    represent the same physical speaker (split by DoA jitter or
    sidelobe riding).  The weaker track is absorbed into the stronger
    one.

    This single step directly prevents the observed "example speaker
    split into two confirmed tracks" failure.

    Parameters
    ----------
    tracks : list[dict]
        Built tracks (with ``doa_track``, ``active_segments``, etc.).
    max_sep_deg : float
        Maximum angular separation to consider a merge.
    min_temporal_iou : float
        Minimum temporal IoU (intersection / union of active frames)
        required to trigger a merge.

    Returns
    -------
    merged_tracks : list[dict]
        Tracks after merging.  Absorbed tracks are removed.
    """
    if len(tracks) < 2:
        return tracks

    # Build active frame sets for each track
    frame_sets: List[set] = []
    for tr in tracks:
        frames = set()
        for f_idx, _az in tr.get("doa_track", []):
            frames.add(int(f_idx))
        frame_sets.append(frames)

    # Score tracks by n_points (primary) and mean_score (tiebreaker)
    strengths = [
        (tr.get("n_points", 0), tr.get("mean_score", 0.0))
        for tr in tracks
    ]

    absorbed: set = set()
    merge_log: List[str] = []

    for i in range(len(tracks)):
        if i in absorbed:
            continue
        for j in range(i + 1, len(tracks)):
            if j in absorbed:
                continue

            sep = _angular_distance(
                tracks[i]["mean_azimuth"], tracks[j]["mean_azimuth"])
            if sep > max_sep_deg:
                continue

            # Temporal IoU
            inter = len(frame_sets[i] & frame_sets[j])
            union = len(frame_sets[i] | frame_sets[j])
            iou = inter / max(union, 1)

            if iou < min_temporal_iou:
                continue

            # Absorb the weaker track
            if strengths[i] >= strengths[j]:
                absorbed.add(j)
                merge_log.append(
                    f"Merge {tracks[j]['mean_azimuth']:.0f}° into "
                    f"{tracks[i]['mean_azimuth']:.0f}° "
                    f"(sep={sep:.0f}°, iou={iou:.2f})")
            else:
                absorbed.add(i)
                merge_log.append(
                    f"Merge {tracks[i]['mean_azimuth']:.0f}° into "
                    f"{tracks[j]['mean_azimuth']:.0f}° "
                    f"(sep={sep:.0f}°, iou={iou:.2f})")
                break  # i is absorbed, stop comparing against others

    if merge_log:
        for msg in merge_log:
            logger.info("  [merge] %s", msg)

    return [tr for i, tr in enumerate(tracks) if i not in absorbed]


# ── 5. Quality filtering ──────────────────────────────────────────────

def filter_tracks(
    tracks: List[Dict[str, Any]],
    n_frames: int,
    min_points_frac: float = 0.03,
    min_duration_s: float = 0.5,
    max_angular_std_deg: float = 30.0,
    burst_score_ratio: float = 0.40,
    burst_abs_floor: float = 0.0,
    burst_percentile: float = 40.0,
    min_group_count: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Classify tracks into confirmed, provisional, and rejected.

    Hard reject:  too few points  /  too short  /  too diffuse.
    Then classify by local burst evidence + pair-group support:

    - **confirmed** : has_burst AND has_group
    - **provisional**: has_burst XOR has_group  (one source of evidence)
    - **rejected**  : weak in both burst AND pair-group

    Burst threshold uses the HIGHER of:
    - an absolute floor (``burst_abs_floor``, default 0.0 = disabled),
    - a percentile-based floor (the ``burst_percentile``-th percentile
      of all tracks' burst values),
    - the legacy ratio-based floor (``burst_score_ratio × best_burst``).

    This prevents weak/intermittent speakers from being suppressed
    simply because they cannot compete with the global best track.

    Returns ``(confirmed, provisional, rejected)``.
    """
    if not tracks:
        return [], [], []

    min_points = max(2, int(min_points_frac * n_frames))

    # Burst reference: compute multiple floors
    all_top3 = [tr.get("mean_top3_window_score_hybrid", 0.0)
                for tr in tracks]
    best_burst = max(all_top3) if all_top3 else 0.0

    # Legacy ratio-based floor
    burst_floor_ratio = burst_score_ratio * best_burst

    # Percentile-based floor (e.g., 40th percentile)
    if len(all_top3) >= 2:
        burst_floor_pctl = float(np.percentile(all_top3, burst_percentile))
    else:
        burst_floor_pctl = 0.0

    # Effective floor: use the LOWEST of ratio and percentile floors
    # to be more permissive, but respect the absolute floor as a minimum
    burst_floor = max(burst_abs_floor, min(burst_floor_ratio, burst_floor_pctl))

    logger.debug("  burst floors: ratio=%.4f pctl=%.4f abs=%.4f -> eff=%.4f",
                 burst_floor_ratio, burst_floor_pctl, burst_abs_floor,
                 burst_floor)

    confirmed: List[Dict[str, Any]] = []
    provisional: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for tr in tracks:
        reason = ""

        # Hard filters
        if tr["n_points"] < min_points:
            reason = f"too_few_points ({tr['n_points']}<{min_points})"
        elif tr["total_duration_s"] < min_duration_s:
            reason = (f"too_short ({tr['total_duration_s']:.2f}s"
                      f"<{min_duration_s:.2f}s)")
        elif max_angular_std_deg > 0:
            astd = tr.get("angular_std_deg", 0.0)
            if astd > max_angular_std_deg:
                reason = (f"too_diffuse (std={astd:.1f}°"
                          f">{max_angular_std_deg:.1f}°)")

        if reason:
            tr["status"] = "rejected"
            tr["rejection_reason"] = reason
            rejected.append(tr)
            logger.info("  REJECT %.0f°: %s", tr["mean_azimuth"], reason)
            continue

        # Soft criteria
        burst_val = tr.get("mean_top3_window_score_hybrid", 0.0)
        group_val = tr.get("support_group_count", 0)
        has_burst = burst_val >= burst_floor
        has_group = group_val >= min_group_count

        if has_burst and has_group:
            tr["status"] = "confirmed"
            confirmed.append(tr)
        elif has_burst or has_group:
            tr["status"] = "provisional"
            note = []
            if not has_burst:
                note.append(f"burst={burst_val:.3f}<{burst_floor:.3f}")
            if not has_group:
                note.append(f"groups={group_val}<{min_group_count}")
            tr["rejection_reason"] = " | ".join(note)
            provisional.append(tr)
        else:
            tr["status"] = "rejected"
            tr["rejection_reason"] = (
                f"weak_both (burst={burst_val:.3f}<{burst_floor:.3f}, "
                f"groups={group_val}<{min_group_count})")
            rejected.append(tr)
            logger.info("  REJECT %.0f°: %s",
                        tr["mean_azimuth"], tr["rejection_reason"])

    # Sort by mean azimuth and assign stable IDs
    confirmed.sort(key=lambda t: t["mean_azimuth"])
    for i, tr in enumerate(confirmed):
        tr["id"] = f"spk{i:02d}"

    provisional.sort(key=lambda t: t["mean_azimuth"])
    for i, tr in enumerate(provisional):
        tr["id"] = f"prov{i:02d}"

    return confirmed, provisional, rejected


# ── 5c. Conversation-cluster detection ────────────────────────────────

def detect_conversation_clusters(
    candidates: List[Dict[str, Any]],
    max_gap_deg: float = 25.0,
) -> List[Dict[str, Any]]:
    """
    Group spatially close candidates into conversation clusters.

    Two candidates belong to the same cluster if they are within
    ``max_gap_deg`` of each other (single-linkage: A-B within gap and
    B-C within gap → A, B, C are one cluster).

    Parameters
    ----------
    candidates : list of track dicts (confirmed + provisional)
        Each must have ``"id"`` and ``"mean_azimuth"``.
    max_gap_deg : float
        Maximum angular separation (degrees) to link two candidates.

    Returns
    -------
    list of cluster dicts, each with:
        - ``cluster_id`` : str, e.g. "conv_cluster_0"
        - ``members`` : list of candidate id strings
        - ``centroid_deg`` : circular mean azimuth of members
        - ``span_deg`` : max angular distance between any pair in cluster
        - ``n_members`` : int
    """
    if not candidates:
        return []

    # Extract (id, azimuth) tuples
    items = []
    for c in candidates:
        cid = c.get("id", "?")
        az = c.get("mean_azimuth", 0.0)
        items.append((cid, az))

    n = len(items)
    # Union-Find for single-linkage clustering
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            gap = _angular_distance(items[i][1], items[j][1])
            if gap <= max_gap_deg:
                union(i, j)

    # Group by root
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clusters: List[Dict[str, Any]] = []
    cluster_idx = 0
    for members_idx in sorted(groups.values(),
                               key=lambda ms: items[ms[0]][1]):
        member_ids = [items[m][0] for m in members_idx]
        azimuths = [items[m][1] for m in members_idx]

        # Circular mean
        rad = np.deg2rad(azimuths)
        centroid = float(np.rad2deg(np.arctan2(
            np.mean(np.sin(rad)), np.mean(np.cos(rad))))) % 360.0

        # Span: max pairwise angular distance
        span = 0.0
        for i2 in range(len(azimuths)):
            for j2 in range(i2 + 1, len(azimuths)):
                span = max(span, _angular_distance(azimuths[i2], azimuths[j2]))

        clusters.append({
            "cluster_id": f"conv_cluster_{cluster_idx}",
            "members": member_ids,
            "centroid_deg": round(centroid, 1),
            "span_deg": round(span, 1),
            "n_members": len(member_ids),
        })
        cluster_idx += 1

    # Annotate each candidate with its cluster_id
    id_to_cluster: Dict[str, str] = {}
    for cl in clusters:
        for mid in cl["members"]:
            id_to_cluster[mid] = cl["cluster_id"]
    for c in candidates:
        c["conversation_cluster"] = id_to_cluster.get(c.get("id", ""), "")

    return clusters


# ── 5b. Residual discovery ────────────────────────────────────────────

def _residual_discovery(
    points: np.ndarray,
    labels: np.ndarray,
    directions: np.ndarray,
    P_hybrid: np.ndarray,
    assign_dist: float,
    min_sep_deg: float = 15.0,
    min_support: int = 50,
    max_total_dirs: int = 14,
    min_hybrid_score: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Find new directions from unassigned (noise) peaks with strict validation.

    Validation checks:
    - Minimum histogram support (point count)
    - Temporal spread (must span a meaningful fraction of the recording)
    - Temporal concentration (burst-like activity, not uniform noise)
    - Hybrid-map energy validation
    - Shoulder rejection: candidates near a stronger existing direction
      must show clearly stronger hybrid evidence to survive

    Returns
    -------
    directions : updated direction array
    labels : updated label array
    n_new : number of new directions added (0 if none)
    """
    if len(directions) >= max_total_dirs:
        return directions, labels, 0

    noise_mask = labels == -1
    n_noise = int(noise_mask.sum())
    if n_noise < min_support:
        return directions, labels, 0

    noise_pts = points[noise_mask]
    noise_az = noise_pts[:, 1]
    noise_frames = noise_pts[:, 0].astype(int)

    # Build azimuth histogram of unassigned peaks + track temporal spread
    n_grid = 360
    hist = np.zeros(n_grid, dtype=np.float64)
    frame_sets: Dict[int, set] = {}  # bin → set of unique frames
    for az, fr in zip(noise_az, noise_frames):
        bin_idx = int(round(az)) % n_grid
        hist[bin_idx] += 1
        frame_sets.setdefault(bin_idx, set()).add(int(fr))

    # Smooth to handle angle jitter
    hist_smooth = gaussian_filter1d(hist, sigma=3, mode='wrap')

    support_thr = max(min_support * 0.4, 8)
    if hist_smooth.max() < support_thr:
        return directions, labels, 0

    # Find peaks in the noise histogram
    pad = int(min_sep_deg) + 2
    tiled = np.concatenate([hist_smooth[-pad:], hist_smooth, hist_smooth[:pad]])
    pks, props = find_peaks(tiled, height=support_thr,
                            distance=int(min_sep_deg))
    pks_orig = pks - pad
    valid = (pks_orig >= 0) & (pks_orig < n_grid)
    pks_orig = pks_orig[valid]
    pk_heights = props["peak_heights"][valid]

    if len(pks_orig) == 0:
        return directions, labels, 0

    # Evaluate and accept candidates
    max_new = max_total_dirs - len(directions)
    new_dirs: List[float] = []
    n_frames_total = P_hybrid.shape[1]

    # Reference for hybrid validation: max of windowed mean energy
    # (burst-aware) instead of global mean.  This is more permissive
    # for intermittent/late speakers whose global average is low but
    # whose best window is clearly above noise.
    win_size = max(1, int(P_hybrid.shape[1] * 0.05))  # ~5% of recording
    step = max(1, win_size // 2)
    frame_max_per_angle = np.zeros(n_grid, dtype=np.float64)
    for a_idx in range(n_grid):
        row = P_hybrid[a_idx, :]
        t = 0
        best_win = 0.0
        while t + win_size <= n_frames_total:
            win_mean = float(row[t:t + win_size].mean())
            if win_mean > best_win:
                best_win = win_mean
            t += step
        frame_max_per_angle[a_idx] = best_win
    hybrid_ref = float(np.percentile(
        frame_max_per_angle[frame_max_per_angle > 0], 75)
    ) if (frame_max_per_angle > 0).any() else float(P_hybrid.max(axis=0).mean())

    # Pre-compute hybrid best-window energy at each existing direction
    # for shoulder check
    existing_hybrid: List[float] = []
    for d in directions:
        d_idx = int(round(d)) % n_grid
        existing_hybrid.append(frame_max_per_angle[d_idx])

    for pk, ht in sorted(zip(pks_orig.tolist(), pk_heights.tolist()),
                         key=lambda x: -x[1]):
        if ht < support_thr or len(new_dirs) >= max_new:
            continue

        # Separation from all existing + new directions
        all_existing = list(directions) + new_dirs
        if any(_angular_distance(float(pk), d) < min_sep_deg
               for d in all_existing):
            continue

        # Temporal spread: count unique frames in ±3° bin range
        unique_frames: set = set()
        for offset in range(-3, 4):
            idx = (pk + offset) % n_grid
            unique_frames.update(frame_sets.get(idx, set()))
        temporal_frac = len(unique_frames) / max(n_frames_total, 1)
        if temporal_frac < 0.02:  # must span ≥2% of recording
            logger.debug("  residual %d°: skip, low temporal spread (%.3f)",
                         pk, temporal_frac)
            continue

        # Temporal concentration: activity should be bursty, not uniform
        # noise.  Check that frames cluster in time (std of frame indices
        # relative to recording length is not too large vs. coverage).
        frame_list = sorted(unique_frames)
        if len(frame_list) >= 5:
            frame_arr = np.array(frame_list, dtype=np.float64)
            frame_iqr = float(np.percentile(frame_arr, 75) -
                              np.percentile(frame_arr, 25))
            # If frames are scattered uniformly, IQR ≈ 0.5 * n_frames.
            # Real speakers tend to have tighter temporal clusters.
            # Reject if IQR spans >80% of recording (uniform noise).
            if frame_iqr > 0.80 * n_frames_total and temporal_frac < 0.05:
                logger.debug("  residual %d°: skip, uniform temporal "
                             "scatter (iqr=%.0f, frac=%.3f)",
                             pk, frame_iqr, temporal_frac)
                continue

        # Validate with hybrid map: best-window energy at this angle
        # (burst-aware, not global mean — intermittent speakers survive)
        az_idx = int(round(pk)) % n_grid
        hybrid_best_win = frame_max_per_angle[az_idx]
        if hybrid_best_win < min_hybrid_score * hybrid_ref:
            logger.debug("  residual %d°: skip, weak hybrid window "
                         "(%.4f < %.4f)",
                         pk, hybrid_best_win, min_hybrid_score * hybrid_ref)
            continue

        # Shoulder rejection: if this candidate sits near (within 2×
        # min_sep_deg) a stronger existing direction, require that its
        # hybrid energy is at least 60% of the neighbour's.  This
        # prevents shoulder lobes from being promoted to directions.
        is_shoulder = False
        for di, d in enumerate(directions):
            sep = _angular_distance(float(pk), d)
            if sep < 2.0 * min_sep_deg:
                neighbour_h = existing_hybrid[di] if di < len(existing_hybrid) else 0.0
                if neighbour_h > 0 and hybrid_best_win < 0.60 * neighbour_h:
                    logger.debug("  residual %d°: skip, shoulder of %.0f° "
                                 "(h=%.4f < 0.60×%.4f)",
                                 pk, d, hybrid_best_win, neighbour_h)
                    is_shoulder = True
                    break
        if is_shoulder:
            continue

        new_dirs.append(float(pk))

    if not new_dirs:
        return directions, labels, 0

    # Expand direction set and re-assign ALL points
    expanded = np.concatenate([directions, np.array(new_dirs)])
    new_labels = assign_peaks_to_directions(
        points, expanded, max_distance_deg=assign_dist,
    )

    return expanded, new_labels, len(new_dirs)


# ── 5c. Scoring helpers ──────────────────────────────────────────────

def _add_hybrid_scores(
    tracks: List[Dict[str, Any]],
    P_hybrid: np.ndarray,
    P_disc: np.ndarray,
) -> None:
    """
    Add ``mean_score_hybrid`` and ``mean_score_disc`` to each track by
    sampling the respective posterior maps at the track's doa_track
    positions.  The original ``mean_score`` (from peak heights during
    extraction) is left untouched.
    """
    n_grid = P_hybrid.shape[0]
    n_frames_h = P_hybrid.shape[1]
    n_frames_d = P_disc.shape[1]

    for tr in tracks:
        doa_track = tr.get("doa_track", [])
        if not doa_track:
            tr["mean_score_hybrid"] = 0.0
            tr["mean_score_disc"] = 0.0
            continue

        h_scores: List[float] = []
        d_scores: List[float] = []
        for frame_idx, az in doa_track:
            f = int(frame_idx)
            a = int(round(az)) % n_grid
            if 0 <= f < n_frames_h:
                h_scores.append(float(P_hybrid[a, f]))
            if 0 <= f < n_frames_d:
                d_scores.append(float(P_disc[a, f]))

        tr["mean_score_hybrid"] = (
            round(float(np.mean(h_scores)), 4) if h_scores else 0.0
        )
        tr["mean_score_disc"] = (
            round(float(np.mean(d_scores)), 4) if d_scores else 0.0
        )


# ── 5d. Second-pass track splitting ─────────────────────────────────

def _second_pass_split(
    points: np.ndarray,
    labels: np.ndarray,
    min_sep_deg: float = 12.0,
    min_points: int = 30,
    max_temporal_overlap: float = 0.70,
) -> Tuple[np.ndarray, List[str]]:
    """
    Second-pass track splitting with temporal validation.

    Tries 2-means on every large-enough cluster.  Accepts the split
    only if:
    1. Both sub-clusters have ≥ ``min_points`` points.
    2. The two centroids are separated by ≥ ``min_sep_deg``.
    3. Temporal overlap between the two child clusters is below
       ``max_temporal_overlap``.  This prevents one broad-peaked
       speaker from being split into two identities.

    Temporal overlap is defined as the fraction of frames active in
    *both* children relative to frames active in *either*.

    Modifies ``labels`` in-place.
    """
    from sklearn.cluster import KMeans

    unique = sorted(set(labels.tolist()) - {-1})
    if not unique:
        return labels, []

    next_label = max(unique) + 1
    split_log: List[str] = []

    for k in unique:
        mask = labels == k
        n_pts = int(mask.sum())
        if n_pts < 2 * min_points:
            continue

        azimuths = points[mask, 1]
        frames_k = points[mask, 0].astype(int)

        # 2-means on unwrapped angle relative to circular mean
        rad = np.deg2rad(azimuths)
        mean_rad = np.arctan2(np.mean(np.sin(rad)),
                              np.mean(np.cos(rad)))
        diff = np.arctan2(np.sin(rad - mean_rad),
                          np.cos(rad - mean_rad))

        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        sub = km.fit_predict(diff.reshape(-1, 1))

        n0, n1 = int((sub == 0).sum()), int((sub == 1).sum())
        if n0 < min_points or n1 < min_points:
            continue

        mean0 = _circular_mean(azimuths[sub == 0])
        mean1 = _circular_mean(azimuths[sub == 1])
        sep = _angular_distance(mean0, mean1)

        if sep < min_sep_deg:
            continue

        # Temporal overlap check: if both children are active in
        # mostly the same frames, it's likely one broad speaker
        frames_0 = set(frames_k[sub == 0].tolist())
        frames_1 = set(frames_k[sub == 1].tolist())
        intersection = len(frames_0 & frames_1)
        union = len(frames_0 | frames_1)
        t_overlap = intersection / max(union, 1)

        if t_overlap > max_temporal_overlap:
            logger.debug("  split reject %.0f°: temporal overlap=%.2f "
                         "(>%.2f)",
                         _circular_mean(azimuths), t_overlap,
                         max_temporal_overlap)
            continue

        # Accept the split
        indices = np.where(mask)[0]
        labels[indices[sub == 1]] = next_label
        next_label += 1

        original_az = _circular_mean(azimuths)
        split_log.append(
            f"{original_az:.0f}°→{mean0:.0f}°+{mean1:.0f}° "
            f"(sep={sep:.0f}°, t_iou={t_overlap:.2f})"
        )

    return labels, split_log


# ── 5e. Local burst statistics ───────────────────────────────────────

def _compute_local_burst_stats(
    tracks: List[Dict[str, Any]],
    P_hybrid: np.ndarray,
    hop_length: int,
    sr: int = 44100,
    window_s: float = 1.0,
) -> None:
    """
    Compute local burst evidence for each track using sliding windows.

    For each ~1 s window, the hybrid score is evaluated only at frames
    where the track is actually active.  This means an intermittent or
    late-starting speaker is judged by their *best local performance*,
    not dragged down by long inactive stretches.

    Adds to each track:
      - max_window_score_hybrid
      - mean_top3_window_score_hybrid
      - active_window_fraction_hybrid
    """
    win_frames = max(1, int(window_s * sr / hop_length))
    step_frames = max(1, win_frames // 2)
    n_grid = P_hybrid.shape[0]
    n_frames = P_hybrid.shape[1]

    for tr in tracks:
        doa_track = tr.get("doa_track", [])
        if not doa_track:
            tr["max_window_score_hybrid"] = 0.0
            tr["mean_top3_window_score_hybrid"] = 0.0
            tr["active_window_fraction_hybrid"] = 0.0
            continue

        # Build frame → azimuth lookup for this track
        frame_az: Dict[int, float] = {int(f): az for f, az in doa_track}

        # Slide windows over entire recording
        window_scores: List[float] = []
        t = 0
        while t + win_frames <= n_frames:
            scores_in: List[float] = []
            for f in range(t, t + win_frames):
                if f in frame_az:
                    a = int(round(frame_az[f])) % n_grid
                    scores_in.append(float(P_hybrid[a, f]))
            if len(scores_in) >= 3:
                window_scores.append(float(np.mean(scores_in)))
            t += step_frames

        if window_scores:
            window_scores.sort(reverse=True)
            tr["max_window_score_hybrid"] = round(window_scores[0], 4)
            top3 = window_scores[:min(3, len(window_scores))]
            tr["mean_top3_window_score_hybrid"] = round(
                float(np.mean(top3)), 4)
            # Fraction of windows with score >= 50% of best
            if window_scores[0] > 0:
                thr = 0.5 * window_scores[0]
                n_above = sum(1 for s in window_scores if s >= thr)
                tr["active_window_fraction_hybrid"] = round(
                    n_above / len(window_scores), 4)
            else:
                tr["active_window_fraction_hybrid"] = 0.0
        else:
            tr["max_window_score_hybrid"] = 0.0
            tr["mean_top3_window_score_hybrid"] = 0.0
            tr["active_window_fraction_hybrid"] = 0.0


# ── 5f. Pair-group support metrics ───────────────────────────────────

def _compute_pair_group_support(
    tracks: List[Dict[str, Any]],
    P_on_ear: Optional[np.ndarray],
    P_diagonal: Optional[np.ndarray],
    P_lateral: Optional[np.ndarray],
    group_threshold_ratio: float = 0.30,
    group_abs_threshold: float = 0.10,
) -> None:
    """
    Add per-track pair-group scores and ``support_group_count``.

    Sidelobes typically appear in only one pair group (e.g. lateral
    pairs cause front/back ghosts), while real speakers are supported
    by multiple groups.  ``support_group_count`` counts how many of
    the 3 groups show meaningful support for this track.

    Per-track evaluation (not global-best comparison)
    --------------------------------------------------
    Instead of comparing each group score against the global best
    across all tracks (which suppresses weak speakers), we use a
    per-track criterion: a group "supports" a track if its score
    exceeds ``group_threshold_ratio × that_track's_best_group_score``
    AND exceeds the absolute floor ``group_abs_threshold``.  This
    treats group support as soft evidence that is independent of how
    strong the globally dominant track is.
    """
    group_maps = [
        ("mean_score_on_ear", P_on_ear),
        ("mean_score_diagonal", P_diagonal),
        ("mean_score_lateral", P_lateral),
    ]

    # First pass: compute per-group mean scores
    for tr in tracks:
        doa_track = tr.get("doa_track", [])
        for key, P_grp in group_maps:
            if P_grp is None or not doa_track:
                tr[key] = 0.0
                continue
            n_grid = P_grp.shape[0]
            scores: List[float] = []
            for frame_idx, az in doa_track:
                f = int(frame_idx)
                a = int(round(az)) % n_grid
                if 0 <= f < P_grp.shape[1]:
                    scores.append(float(P_grp[a, f]))
            tr[key] = round(float(np.mean(scores)), 4) if scores else 0.0

    # Second pass: per-track support_group_count
    for tr in tracks:
        group_scores = [tr.get(key, 0.0) for key, _ in group_maps]
        best_own_group = max(group_scores) if group_scores else 0.0
        threshold = max(group_abs_threshold,
                        group_threshold_ratio * best_own_group)

        count = 0
        for gs in group_scores:
            if gs >= threshold:
                count += 1
        tr["support_group_count"] = count


# ── 6. Example-aware validation ───────────────────────────────────────

EXAMPLE_EXPECTED_DIRS = [0.0, 90.0, 180.0, 270.0]


def _validate_example_tracks(tracks: List[Dict[str, Any]]) -> None:
    """
    Compare final tracks against the four known example directions and
    log angular errors.  Purely informational — does not modify tracks.
    """
    detected = [t["mean_azimuth"] for t in tracks]

    # Per-direction error logging at DEBUG level
    for exp in EXAMPLE_EXPECTED_DIRS:
        if not detected:
            logger.debug("    %3.0f° expected -> NO MATCH", exp)
            continue
        dists = [_angular_distance(exp, d) for d in detected]
        best_idx = int(np.argmin(dists))
        logger.debug("    %3.0f° -> %5.1f°  (err=%.1f°)",
                     exp, detected[best_idx], dists[best_idx])

    # Compact summary at INFO
    if len(detected) == len(EXAMPLE_EXPECTED_DIRS):
        errors = []
        used = set()
        for exp in EXAMPLE_EXPECTED_DIRS:
            dists = [(i, _angular_distance(exp, d))
                     for i, d in enumerate(detected) if i not in used]
            best_i, best_d = min(dists, key=lambda x: x[1])
            errors.append(best_d)
            used.add(best_i)
        logger.info("[example][track] mean_err=%.1f° | detected=%s",
                    np.mean(errors),
                    [f"{d:.1f}" for d in detected])
    else:
        logger.warning("[example][track] count mismatch: got %d, "
                       "expected %d | detected=%s",
                       len(detected), len(EXAMPLE_EXPECTED_DIRS),
                       [f"{d:.1f}" for d in detected])


# ── Entry point ────────────────────────────────────────────────────────

def main(tag: str = "mixture") -> None:
    """
    Run direction tracking for the given tag.

    Pipeline:
      1. Global + windowed direction discovery  (on P_disc = SRP-norm)
      2. Per-frame peak extraction              (on P_extract)
      3. Assign peaks → directions → residual → split → tracks
      4. Score tracks: hybrid, disc, burst, pair-group
      5. Filter: confirmed / provisional / rejected
      6. Save confirmed-only final JSON + full raw debug JSON
    """
    ensure_output_dirs()

    doa_cfg = CFG.get("doa", {})

    # ── Config parameters ─────────────────────────────────────────────
    top_k        = doa_cfg.get("top_k_peaks_per_frame", 4)
    min_dist     = doa_cfg.get("min_peak_distance_deg", 25)
    rel_thr      = doa_cfg.get("peak_rel_threshold", 0.15)
    second_ratio = doa_cfg.get("second_peak_ratio", 0.30)
    assign_dist  = doa_cfg.get("max_assign_dist_deg", 20.0)
    prominence   = doa_cfg.get("global_peak_prominence", 0.10)
    smooth_sigma = doa_cfg.get("smooth_track_sigma", 5)
    min_pt_frac  = doa_cfg.get("min_track_points_frac", 0.03)
    min_dur      = doa_cfg.get("min_track_duration_s", 0.5)

    # Windowed discovery
    use_windowed = doa_cfg.get("use_windowed_discovery", True)
    win_dur_s    = float(doa_cfg.get("window_duration_s", 1.5))
    win_overlap  = float(doa_cfg.get("window_overlap_frac", 0.5))
    win_min_cnt  = int(doa_cfg.get("windowed_peak_min_count", 3))

    # Track splitting
    do_split     = doa_cfg.get("split_broad_tracks", True)
    split_std    = float(doa_cfg.get("track_split_std_deg", 18.0))
    split_min_pt = int(doa_cfg.get("track_split_min_points", 30))

    # Diffuseness rejection
    max_ang_std  = float(doa_cfg.get("reject_diffuse_std_deg", 30.0))

    # Dual-map tracking
    use_dual_map = bool(doa_cfg.get("use_dual_map_tracking", False))

    # Discovery thresholds
    disc_rel_thr   = float(doa_cfg.get("discovery_global_rel_threshold", rel_thr))
    disc_prom      = float(doa_cfg.get("discovery_global_prominence", prominence))
    disc_max_glob  = int(doa_cfg.get("discovery_max_global_dirs", 10))
    disc_win_rel   = float(doa_cfg.get("windowed_rel_threshold", rel_thr))
    disc_max_win   = int(doa_cfg.get("discovery_max_windowed_dirs", 6))
    disc_merge_sep = float(doa_cfg.get("discovery_merge_sep_deg", assign_dist))

    # Residual discovery
    use_residual    = bool(doa_cfg.get("use_residual_discovery", False))
    resid_passes    = int(doa_cfg.get("residual_num_passes", 2))
    resid_min_sup   = int(doa_cfg.get("residual_min_support_frames", 50))
    resid_merge_sep = float(doa_cfg.get("residual_merge_sep_deg", 15.0))
    max_total_dirs  = int(doa_cfg.get("discovery_max_total_dirs", 14))

    # Second-pass track splitting
    do_2nd_split    = bool(doa_cfg.get("track_second_pass_split", False))
    split2_sep      = float(doa_cfg.get("track_second_pass_min_sep_deg", 15.0))

    # Smart filtering
    burst_win_s     = float(doa_cfg.get("burst_window_s", 1.0))
    burst_sc_ratio  = float(doa_cfg.get("burst_score_ratio", 0.40))
    burst_abs_fl    = float(doa_cfg.get("burst_abs_floor", 0.0))
    burst_pctl      = float(doa_cfg.get("burst_percentile", 40.0))
    min_grp_cnt     = int(doa_cfg.get("min_group_count", 2))
    grp_thr_ratio   = float(doa_cfg.get("group_threshold_ratio", 0.30))
    grp_abs_thr     = float(doa_cfg.get("group_abs_threshold", 0.10))

    # Track merging — anti over-finding
    merge_sep       = float(doa_cfg.get("merge_max_sep_deg", 12.0))
    merge_iou       = float(doa_cfg.get("merge_min_temporal_iou", 0.60))

    # Wide-angle sidelobe merge
    wide_merge_sep  = float(doa_cfg.get("wide_merge_max_sep_deg", 40.0))
    wide_merge_iou  = float(doa_cfg.get("wide_merge_min_temporal_iou", 0.50))

    # Provisional cluster consolidation
    prov_cluster_gap  = float(doa_cfg.get("provisional_cluster_gap_deg", 40.0))
    prov_cluster_min  = int(doa_cfg.get("provisional_cluster_min_size", 3))
    prov_cluster_keep = int(doa_cfg.get("provisional_cluster_max_keep", 1))

    # Temporal persistence for peak extraction
    persist_win     = int(doa_cfg.get("temporal_persist_window", 5))
    persist_cnt     = int(doa_cfg.get("temporal_persist_count", 3))
    persist_tol     = float(doa_cfg.get("temporal_persist_tol_deg", 8.0))

    # Pre-split temporal overlap gate
    presplit_max_overlap = float(doa_cfg.get(
        "presplit_max_temporal_overlap", 0.70))

    stft_params = get_stft_params()
    hop_length  = stft_params.get("hop_length", 256)

    # ── Load angular power maps from step 02 ──────────────────────────
    if use_dual_map:
        _srp_p = DOA_DIR / f"{tag}_doa_posteriors_srp_norm.npy"
        _hyb_p = DOA_DIR / f"{tag}_doa_posteriors_hybrid.npy"
        if _srp_p.exists() and _hyb_p.exists():
            P_disc = np.load(str(_srp_p))
            P_score = np.load(str(_hyb_p))
            logger.info("[%s] Dual maps loaded: disc=%s  score=%s",
                        tag, P_disc.shape, P_score.shape)
        else:
            logger.warning("[%s] Dual-map files missing — single-map fallback",
                           tag)
            use_dual_map = False

    if not use_dual_map:
        post_path = DOA_DIR / f"{tag}_doa_posteriors.npy"
        if not post_path.exists():
            raise FileNotFoundError(
                f"DoA posteriors not found: {post_path}  "
                "-- run step 02 first.")
        P_disc = np.load(str(post_path))
        P_score = P_disc
        logger.info("[%s] Single-map loaded: %s", tag, P_disc.shape)

    n_grid, n_frames = P_disc.shape

    # ── Load additional maps (optional — graceful fallback) ───────────
    def _load_opt(suffix: str) -> Optional[np.ndarray]:
        p = DOA_DIR / f"{tag}_doa_posteriors_{suffix}.npy"
        return np.load(str(p)) if p.exists() else None

    P_extract  = _load_opt("extract")
    P_on_ear   = _load_opt("on_ear")
    P_diagonal = _load_opt("diagonal")
    P_lateral  = _load_opt("lateral")

    if P_extract is not None:
        logger.info("[%s] Loaded extract + pair-group maps", tag)

    # ── Step 1: global direction discovery (on P_disc) ────────────────
    global_dirs, global_hts = find_global_directions(
        P_disc,
        min_distance_deg=min_dist,
        rel_threshold=disc_rel_thr,
        prominence_frac=disc_prom,
        max_speakers=disc_max_glob,
    )
    logger.info("[%s] Global dirs (%d): %s", tag, len(global_dirs),
                [f"{d:.0f}°" for d in global_dirs])

    # ── Step 1b: windowed discovery (on P_disc) ───────────────────────
    if use_windowed:
        win_dirs, win_hts = find_windowed_directions(
            P_disc,
            hop_length=hop_length,
            sr=SAMPLE_RATE,
            window_dur_s=win_dur_s,
            window_overlap=win_overlap,
            min_distance_deg=min_dist,
            rel_threshold=disc_win_rel,
            min_window_count=win_min_cnt,
            max_new_dirs=disc_max_win,
        )
        if len(win_dirs) > 0:
            logger.info("[%s] Windowed dirs (%d): %s", tag, len(win_dirs),
                        [f"{d:.0f}°" for d in win_dirs])
    else:
        win_dirs = np.array([], dtype=np.float64)
        win_hts = np.array([], dtype=np.float64)

    # ── Step 1c: merge direction sets ─────────────────────────────────
    directions, dir_heights, dir_sources = _merge_direction_sets(
        global_dirs, global_hts, win_dirs, win_hts,
        min_sep_deg=disc_merge_sep,
    )
    n_global = sum(1 for s in dir_sources if s == "global")
    n_windowed = sum(1 for s in dir_sources if s == "windowed")
    n_residual = 0
    logger.info("[%s] Merged dirs: %d (%dG+%dW) = %s",
                tag, len(directions), n_global, n_windowed,
                [f"{d:.0f}°" for d in directions])

    # ── Step 2: per-frame peak extraction (on P_extract) ──────────────
    P_peak = P_extract if P_extract is not None else P_disc
    points = extract_peaks_per_frame(
        P_peak,
        top_k=top_k,
        min_distance_deg=min_dist,
        rel_threshold=rel_thr,
        second_peak_ratio=second_ratio,
        temporal_persist_window=persist_win,
        temporal_persist_count=persist_cnt,
        temporal_persist_tol_deg=persist_tol,
    )
    logger.info("[%s] Extracted %d peak points (top_k=%d, gate=%.0f%%)",
                tag, points.shape[0], top_k, second_ratio * 100)

    n_peak_points = int(points.shape[0])

    if n_peak_points == 0 or len(directions) == 0:
        logger.warning("[%s] No peaks or no directions — writing empty.", tag)
        tracks: List[Dict[str, Any]] = []
        confirmed: List[Dict[str, Any]] = []
        provisional: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
    else:
        # ── Step 3: assign peaks to directions ────────────────────────
        labels = assign_peaks_to_directions(
            points, directions, max_distance_deg=assign_dist,
        )
        n_assigned = int((labels >= 0).sum())
        n_noise = int((labels == -1).sum())
        logger.info("[%s] Assignment: %d assigned, %d noise",
                    tag, n_assigned, n_noise)

        # ── Step 3a: residual discovery (validated with P_score) ──────
        if use_residual:
            for pass_i in range(resid_passes):
                if len(directions) >= max_total_dirs:
                    break
                directions, labels, n_new = _residual_discovery(
                    points, labels, directions,
                    P_score,
                    assign_dist=assign_dist,
                    min_sep_deg=resid_merge_sep,
                    min_support=resid_min_sup,
                    max_total_dirs=max_total_dirs,
                )
                if n_new == 0:
                    break
                n_residual += n_new
                logger.info("[%s] Residual pass %d: +%d → %d dirs",
                            tag, pass_i + 1, n_new, len(directions))
            if n_residual > 0:
                logger.info("[%s] Dirs after residual: %d "
                            "(%dG+%dW+%dR) = %s",
                            tag, len(directions), n_global, n_windowed,
                            n_residual,
                            [f"{d:.0f}°" for d in directions])

        # ── Step 3b: first-pass split (high-std trigger) ──────────────
        if do_split:
            labels, split_log = _presplit_broad_clusters(
                points, labels,
                split_std_deg=split_std,
                split_min_points=split_min_pt,
                max_temporal_overlap=presplit_max_overlap,
            )
            if split_log:
                logger.info("[%s] Splits (1st): %s", tag,
                            " | ".join(split_log))

        # ── Step 3c: second-pass split (separation-based) ────────────
        if do_2nd_split:
            labels, split_log2 = _second_pass_split(
                points, labels,
                min_sep_deg=split2_sep,
                min_points=split_min_pt,
            )
            if split_log2:
                logger.info("[%s] Splits (2nd): %s", tag,
                            " | ".join(split_log2))

        # ── Step 4: build tracks ──────────────────────────────────────
        tracks = build_tracks(
            points, labels, n_frames,
            hop_length=hop_length,
            smooth_sigma=smooth_sigma,
        )
        n_raw = len(tracks)
        logger.info("[%s] Built %d raw tracks", tag, n_raw)

        # ── Step 4b: score tracks (hybrid + disc) ─────────────────────
        _add_hybrid_scores(tracks, P_score, P_disc)

        # ── Step 4c: local burst statistics ───────────────────────────
        _compute_local_burst_stats(
            tracks, P_score, hop_length, SAMPLE_RATE,
            window_s=burst_win_s,
        )

        # ── Step 4d: pair-group support ───────────────────────────────
        _compute_pair_group_support(
            tracks, P_on_ear, P_diagonal, P_lateral,
            group_threshold_ratio=grp_thr_ratio,
            group_abs_threshold=grp_abs_thr,
        )

        # ── Step 4e: merge close & overlapping tracks ─────────────────
        n_pre_merge = len(tracks)
        tracks = merge_tracks_if_close_and_overlapping(
            tracks,
            max_sep_deg=merge_sep,
            min_temporal_iou=merge_iou,
        )
        if len(tracks) < n_pre_merge:
            logger.info("[%s] Merged %d → %d tracks (anti-split)",
                        tag, n_pre_merge, len(tracks))

        # ── Step 4f: wide-angle merge (sidelobes & reflections) ───────
        if wide_merge_sep > merge_sep:
            n_pre_wide = len(tracks)
            tracks = merge_tracks_if_close_and_overlapping(
                tracks,
                max_sep_deg=wide_merge_sep,
                min_temporal_iou=wide_merge_iou,
            )
            if len(tracks) < n_pre_wide:
                logger.info("[%s] Wide merge: %d → %d tracks "
                            "(sidelobe/reflection removal)",
                            tag, n_pre_wide, len(tracks))

        # ── Step 5: smart filtering ───────────────────────────────────
        confirmed, provisional, rejected = filter_tracks(
            tracks, n_frames,
            min_points_frac=min_pt_frac,
            min_duration_s=min_dur,
            max_angular_std_deg=max_ang_std,
            burst_score_ratio=burst_sc_ratio,
            burst_abs_floor=burst_abs_fl,
            burst_percentile=burst_pctl,
            min_group_count=min_grp_cnt,
        )
        logger.info("[%s] Filter: %d confirmed, %d provisional, "
                    "%d rejected (of %d raw)",
                    tag, len(confirmed), len(provisional),
                    len(rejected), n_raw)

    # ── Step 5b: provisional cluster consolidation ────────────────────
    # When ≥N provisionals chain-link into a single sidelobe fan,
    # keep only the best one(s) per cluster.
    if len(provisional) >= prov_cluster_min:
        prov_clusters = detect_conversation_clusters(
            provisional, max_gap_deg=prov_cluster_gap)
        new_prov: List[Dict[str, Any]] = []
        for cl in prov_clusters:
            cl_ids = set(cl["members"])
            cl_tracks = [p for p in provisional if p.get("id") in cl_ids]
            if len(cl_tracks) >= prov_cluster_min:
                # Sort by mean_score descending, keep top N
                cl_tracks.sort(
                    key=lambda t: t.get("mean_score", 0), reverse=True)
                kept = cl_tracks[:prov_cluster_keep]
                dropped = cl_tracks[prov_cluster_keep:]
                for d in dropped:
                    d["status"] = "rejected"
                    d["rejection_reason"] = "prov_cluster_thinned"
                    rejected.append(d)
                new_prov.extend(kept)
                logger.info(
                    "[%s] Prov cluster %s (%d members): keep %s, "
                    "drop %s",
                    tag, cl["cluster_id"], len(cl_tracks),
                    [t["id"] for t in kept],
                    [t["id"] for t in dropped])
            else:
                new_prov.extend(cl_tracks)
        if len(new_prov) < len(provisional):
            logger.info("[%s] Provisional thinning: %d → %d",
                        tag, len(provisional), len(new_prov))
            provisional = new_prov
            # Re-assign stable IDs after thinning
            provisional.sort(key=lambda t: t["mean_azimuth"])
            for i, tr in enumerate(provisional):
                tr["id"] = f"prov{i:02d}"

    # ── Raw debug JSON (all tracks with full metrics) ───────────────────
    raw_entries: List[Dict[str, Any]] = []
    for tr_list in (confirmed, provisional, rejected):
        for tr in tr_list:
            entry = {k: v for k, v in tr.items()
                     if k not in ("azimuth_track", "doa_track")}
            raw_entries.append(entry)
    raw_entries.sort(key=lambda e: e.get("mean_azimuth", 0))

    # Direction source list
    dir_info: List[Dict[str, Any]] = []
    for i, d in enumerate(directions):
        src = dir_sources[i] if i < len(dir_sources) else "residual"
        dir_info.append({"azimuth_deg": round(float(d), 1), "source": src})

    raw_debug = {
        "tag": tag,
        "n_frames": int(n_frames),
        "discovered_directions": dir_info,
        "n_raw_tracks": len(raw_entries),
        "n_confirmed": len(confirmed),
        "n_provisional": len(provisional),
        "n_rejected": len(rejected),
        "tracks": raw_entries,
    }
    raw_path = DOA_DIR / f"{tag}_doa_tracks_raw.json"
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(raw_debug, fh, indent=2)

    # ── Compact summary ─────────────────────────────────────────────────
    all_kept = confirmed + provisional
    if confirmed:
        az_list  = [f"{t['mean_azimuth']:.1f}" for t in confirmed]
        dur_list = [f"{t['total_duration_s']:.1f}" for t in confirmed]
        h_list   = [f"{t.get('mean_score_hybrid', 0):.3f}" for t in confirmed]
        b_list   = [f"{t.get('mean_top3_window_score_hybrid', 0):.3f}"
                    for t in confirmed]
        g_list   = [f"{t.get('support_group_count', 0)}" for t in confirmed]
        logger.info("[%s][track] confirmed=%d | az=[%s] | dur=[%s] | "
                    "h=[%s] | burst=[%s] | grp=[%s]",
                    tag, len(confirmed),
                    ",".join(az_list), ",".join(dur_list),
                    ",".join(h_list), ",".join(b_list),
                    ",".join(g_list))
    else:
        logger.info("[%s][track] confirmed=0", tag)

    if provisional:
        prov_az = [f"{t['mean_azimuth']:.1f}" for t in provisional]
        logger.info("[%s][track] provisional=%d | az=[%s]",
                    tag, len(provisional), ",".join(prov_az))

    # ── Step 5c: conversation-cluster detection ──────────────────────
    conv_clusters = detect_conversation_clusters(
        confirmed + provisional, max_gap_deg=25.0)
    multi_member = [c for c in conv_clusters if c["n_members"] > 1]
    if multi_member:
        for cl in multi_member:
            logger.info("[%s][cluster] %s: members=%s  centroid=%.1f°  span=%.1f°",
                        tag, cl["cluster_id"],
                        cl["members"], cl["centroid_deg"], cl["span_deg"])
    else:
        logger.info("[%s][cluster] no multi-member conversation clusters found", tag)

    # ── Step 6: example validation ────────────────────────────────────
    if tag == "example" and all_kept:
        _validate_example_tracks(all_kept)

    # ── Save final output ─────────────────────────────────────────────
    result = {
        "tag": tag,
        "n_frames": int(n_frames),
        "n_grid": int(n_grid),
        "hop_length": hop_length,
        "sample_rate": SAMPLE_RATE,
        "n_peak_points": n_peak_points,
        "n_confirmed": len(confirmed),
        "n_provisional": len(provisional),
        "n_rejected": len(rejected),
        "conversation_clusters": conv_clusters,
        "tracking_params": {
            "method": "global_direction_assignment",
            "n_global_directions": n_global,
            "n_windowed_directions": n_windowed,
            "n_residual_directions": n_residual,
            "n_total_directions": len(directions),
            "use_dual_map_tracking": use_dual_map,
            "use_windowed_discovery": use_windowed,
            "use_residual_discovery": use_residual,
            "track_second_pass_split": do_2nd_split,
            "split_broad_tracks": do_split,
            "max_assign_dist_deg": assign_dist,
            "second_peak_ratio": second_ratio,
            "min_track_points_frac": min_pt_frac,
            "min_track_duration_s": min_dur,
            "reject_diffuse_std_deg": max_ang_std,
            "burst_score_ratio": burst_sc_ratio,
            "min_group_count": min_grp_cnt,
        },
        "candidates": confirmed,
        "provisional_candidates": provisional,
    }

    tag_path = DOA_DIR / f"{tag}_doa_tracks.json"
    with open(tag_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    if tag == "mixture":
        canonical_path = DOA_DIR / "doa_tracks.json"
        with open(canonical_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    logger.info("[%s][track] saved → %s  (+ raw: %s)", tag,
                tag_path.name, raw_path.name)


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--tag", default="mixture",
                    help="Input tag (default: mixture)")
    main(_p.parse_args().tag)
