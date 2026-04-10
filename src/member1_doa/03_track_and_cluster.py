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
from typing import Any, Dict, List, Tuple

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
) -> np.ndarray:
    """
    Extract per-frame peaks with a second-peak gate.

    The strongest peak in each frame is always kept (if it exceeds
    ``rel_threshold``).  Additional peaks are kept only if their height
    is ≥ ``second_peak_ratio × strongest_peak_in_frame``.

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

    Returns
    -------
    points : np.ndarray, shape (N, 3)
        Each row is ``(frame_idx, azimuth_deg, score)``.
    """
    n_grid, n_frames = P.shape
    all_points: List[Tuple[int, float, float]] = []

    for t in range(n_frames):
        spectrum = P[:, t]
        frame_max = spectrum.max()
        if frame_max < 1e-8:
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
            continue

        # Sort by height (strongest first)
        order = np.argsort(heights)[::-1][:top_k]
        best_height = heights[order[0]]

        for rank, idx in enumerate(order):
            h = heights[idx]
            # Always accept the strongest; gate the rest
            if rank > 0 and h < second_peak_ratio * best_height:
                continue
            all_points.append((t, float(peaks_orig[idx]), float(h)))

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
) -> Tuple[np.ndarray, List[str]]:
    """
    Inspect each cluster of assigned peaks.  If a cluster's angular
    spread is wider than ``split_std_deg``, attempt to split it into
    two sub-clusters using 2-means on the unwrapped angle.

    The split is only accepted if both sub-clusters have at least
    ``split_min_points`` points and are at least 10° apart.  This
    prevents spurious splits of noisy but unimodal clusters.

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

        # Accept the split: reassign sub-cluster 1 to a new label
        indices = np.where(mask)[0]
        labels[indices[sub == 1]] = next_label
        next_label += 1

        original_az = _circular_mean(azimuths)
        split_log.append(
            f"{original_az:.0f}°→{mean0:.0f}°+{mean1:.0f}° (sep={sep:.0f}°)"
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

        # Restrict to observed activity range
        f_min, f_max = int(frames.min()), int(frames.max())
        track_az = np.full(n_frames, np.nan)
        track_az[f_min:f_max + 1] = smoothed[f_min:f_max + 1]

        # Build active segments (merge small gaps)
        active_frames = np.sort(np.unique(frames))
        segments: List[List[float]] = []
        if len(active_frames) > 0:
            seg_start = active_frames[0]
            prev = active_frames[0]
            for fi in active_frames[1:]:
                if fi - prev > segment_merge_gap:
                    t0 = round(float(seg_start * hop_length / sr), 3)
                    t1 = round(float(prev * hop_length / sr), 3)
                    if t1 - t0 >= min_segment_dur_s:
                        segments.append([t0, t1])
                    seg_start = fi
                prev = fi
            t0 = round(float(seg_start * hop_length / sr), 3)
            t1 = round(float(prev * hop_length / sr), 3)
            if t1 - t0 >= min_segment_dur_s:
                segments.append([t0, t1])

        if not segments:
            continue

        total_dur = sum(s[1] - s[0] for s in segments)

        # doa_track: [[frame_idx, azimuth_deg], ...]
        doa_track: List[List[float]] = [
            [f_idx, round(float(track_az[f_idx]), 1)]
            for f_idx in range(n_frames) if not np.isnan(track_az[f_idx])
        ]

        tracks.append({
            "id": "",                         # assigned after filtering
            "mean_azimuth": round(mean_az, 1),
            "mean_score": round(mean_score, 4),
            "n_points": int(mask.sum()),
            "angular_std_deg": round(angular_std, 1),
            "total_duration_s": round(total_dur, 3),
            "doa_track": doa_track,
            "active_segments": segments,
            "azimuth_track": [
                round(float(v), 1) if not np.isnan(v) else None
                for v in track_az
            ],
        })

    return tracks


# ── 5. Quality filtering ──────────────────────────────────────────────

def filter_tracks(
    tracks: List[Dict[str, Any]],
    n_frames: int,
    min_points_frac: float = 0.03,
    min_duration_s: float = 0.5,
    min_score_ratio: float = 0.15,
    max_angular_std_deg: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Remove weak / short / ghost / diffuse tracks.

    Criteria (a track is **removed** if *any* condition is true):
      1. n_points  <  min_points_frac × n_frames
      2. total active duration  <  min_duration_s
      3. mean_score  <  min_score_ratio × best_track_mean_score
      4. angular_std_deg  >  max_angular_std_deg  (too diffuse)

    Parameters
    ----------
    tracks : list[dict]
    n_frames : int
    min_points_frac : float
    min_duration_s : float
    min_score_ratio : float
    max_angular_std_deg : float
        Reject tracks whose assigned-points angular std exceeds this.
        Set to 0 or negative to disable.

    Returns
    -------
    filtered : list[dict]
        Surviving tracks, re-numbered as spk00, spk01, …
    """
    if not tracks:
        return []

    min_points = max(2, int(min_points_frac * n_frames))
    best_score = max(t["mean_score"] for t in tracks)
    score_floor = min_score_ratio * best_score

    kept: List[Dict[str, Any]] = []
    for tr in tracks:
        if tr["n_points"] < min_points:
            logger.debug("  REJECT %s: too few points (%d < %d)",
                         tr["mean_azimuth"], tr["n_points"], min_points)
            continue
        if tr["total_duration_s"] < min_duration_s:
            logger.debug("  REJECT %s: too short (%.2fs < %.2fs)",
                         tr["mean_azimuth"], tr["total_duration_s"],
                         min_duration_s)
            continue
        if tr["mean_score"] < score_floor:
            logger.debug("  REJECT %s: too weak (%.4f < %.4f)",
                         tr["mean_azimuth"], tr["mean_score"], score_floor)
            continue
        # Angular diffuseness check
        if max_angular_std_deg > 0:
            astd = tr.get("angular_std_deg", 0.0)
            if astd > max_angular_std_deg:
                logger.info("  REJECT %.0f°: too diffuse (std=%.1f° > %.1f°)",
                            tr["mean_azimuth"], astd, max_angular_std_deg)
                continue
        kept.append(tr)

    # Sort by mean azimuth and assign stable IDs
    kept.sort(key=lambda t: t["mean_azimuth"])
    for i, tr in enumerate(kept):
        tr["id"] = f"spk{i:02d}"

    return kept


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

    Parameters
    ----------
    tag : str
        Input tag, e.g. ``"example"`` or ``"mixture"``.
    """
    ensure_output_dirs()

    doa_cfg = CFG.get("doa", {})

    # ── Config parameters ─────────────────────────────────────────────
    top_k        = doa_cfg.get("top_k_peaks_per_frame", 3)
    min_dist     = doa_cfg.get("min_peak_distance_deg", 40)
    rel_thr      = doa_cfg.get("peak_rel_threshold", 0.15)
    second_ratio = doa_cfg.get("second_peak_ratio", 0.50)
    assign_dist  = doa_cfg.get("max_assign_dist_deg", 25.0)
    prominence   = doa_cfg.get("global_peak_prominence", 0.10)
    smooth_sigma = doa_cfg.get("smooth_track_sigma", 5)
    min_pt_frac  = doa_cfg.get("min_track_points_frac", 0.03)
    min_dur      = doa_cfg.get("min_track_duration_s", 0.5)
    min_sc_ratio = doa_cfg.get("min_track_score_ratio", 0.15)

    # Windowed discovery config
    use_windowed = doa_cfg.get("use_windowed_discovery", True)
    win_dur_s    = float(doa_cfg.get("window_duration_s", 1.5))
    win_overlap  = float(doa_cfg.get("window_overlap_frac", 0.5))
    win_min_cnt  = int(doa_cfg.get("windowed_peak_min_count", 3))

    # Track splitting config
    do_split     = doa_cfg.get("split_broad_tracks", True)
    split_std    = float(doa_cfg.get("track_split_std_deg", 18.0))
    split_min_pt = int(doa_cfg.get("track_split_min_points", 30))

    # Diffuseness rejection
    max_ang_std  = float(doa_cfg.get("reject_diffuse_std_deg", 30.0))

    stft_params = get_stft_params()
    hop_length  = stft_params.get("hop_length", 256)

    # ── Load angular power map from step 02 ───────────────────────────
    post_path = DOA_DIR / f"{tag}_doa_posteriors.npy"
    if not post_path.exists():
        raise FileNotFoundError(
            f"DoA posteriors not found: {post_path}  -- run step 02 first."
        )

    P = np.load(str(post_path))
    n_grid, n_frames = P.shape
    logger.info("[%s] Loaded posteriors: %s", tag, P.shape)

    # ── Step 1: find global dominant directions ───────────────────────
    global_dirs, global_hts = find_global_directions(
        P,
        min_distance_deg=min_dist,
        rel_threshold=rel_thr,
        prominence_frac=prominence,
    )
    logger.debug("[%s] Global directions: %s", tag,
                 [f"{d:.0f}°" for d in global_dirs])

    # ── Step 1b: windowed discovery ───────────────────────────────────
    if use_windowed:
        win_dirs, win_hts = find_windowed_directions(
            P,
            hop_length=hop_length,
            sr=SAMPLE_RATE,
            window_dur_s=win_dur_s,
            window_overlap=win_overlap,
            min_distance_deg=min_dist,
            rel_threshold=rel_thr,
            min_window_count=win_min_cnt,
        )
        if len(win_dirs) > 0:
            logger.info("[%s] Windowed discovery: %s", tag,
                        [f"{d:.0f}°" for d in win_dirs])
    else:
        win_dirs = np.array([], dtype=np.float64)
        win_hts = np.array([], dtype=np.float64)

    # ── Step 1c: merge direction sets ─────────────────────────────────
    directions, dir_heights, dir_sources = _merge_direction_sets(
        global_dirs, global_hts, win_dirs, win_hts,
        min_sep_deg=assign_dist,
    )
    n_global = sum(1 for s in dir_sources if s == "global")
    n_windowed = sum(1 for s in dir_sources if s == "windowed")
    logger.info("[%s] Directions: %d total (%d global + %d windowed) = %s",
                tag, len(directions), n_global, n_windowed,
                [f"{d:.0f}°" for d in directions])

    # ── Step 2: selective per-frame peak extraction ───────────────────
    logger.info("[%s] Extracting peaks (top_k=%d, gate=%.0f%%) ...",
                tag, top_k, second_ratio * 100)
    points = extract_peaks_per_frame(
        P,
        top_k=top_k,
        min_distance_deg=min_dist,
        rel_threshold=rel_thr,
        second_peak_ratio=second_ratio,
    )
    logger.debug("[%s] Extracted %d peak points.", tag, points.shape[0])

    if points.shape[0] == 0 or len(directions) == 0:
        logger.warning("[%s] No peaks or no directions -- writing empty.", tag)
        tracks: List[Dict[str, Any]] = []
    else:
        # ── Step 3: assign peaks to directions ────────────────────────
        labels = assign_peaks_to_directions(
            points, directions, max_distance_deg=assign_dist,
        )
        n_assigned = int((labels >= 0).sum())
        n_noise = int((labels == -1).sum())
        logger.debug("[%s] Assigned %d points, %d noise.",
                     tag, n_assigned, n_noise)

        # ── Step 3b: pre-split broad clusters ─────────────────────────
        if do_split:
            labels, split_log = _presplit_broad_clusters(
                points, labels,
                split_std_deg=split_std,
                split_min_points=split_min_pt,
            )
            if split_log:
                logger.info("[%s] Splits: %s", tag,
                            " | ".join(split_log))

        # ── Step 4: build tracks ──────────────────────────────────────
        tracks = build_tracks(
            points, labels, n_frames,
            hop_length=hop_length,
            smooth_sigma=smooth_sigma,
        )
        logger.debug("[%s] Built %d raw tracks.", tag, len(tracks))

        # ── Step 5: quality filtering ─────────────────────────────────
        tracks = filter_tracks(
            tracks, n_frames,
            min_points_frac=min_pt_frac,
            min_duration_s=min_dur,
            min_score_ratio=min_sc_ratio,
            max_angular_std_deg=max_ang_std,
        )
        logger.debug("[%s] After filtering: %d track(s).", tag, len(tracks))

    # ── Compact summary line ────────────────────────────────────────────
    if tracks:
        az_list = [f"{t['mean_azimuth']:.1f}" for t in tracks]
        dur_list = [f"{t['total_duration_s']:.1f}" for t in tracks]
        sc_list = [f"{t['mean_score']:.2f}" for t in tracks]
        std_list = [f"{t.get('angular_std_deg', 0):.0f}" for t in tracks]
        logger.info("[%s][track] n=%d | az=[%s] | dur=[%s] | "
                    "score=[%s] | std=[%s]",
                    tag, len(tracks),
                    ",".join(az_list), ",".join(dur_list),
                    ",".join(sc_list), ",".join(std_list))
    else:
        logger.info("[%s][track] n=0 (no tracks survived filtering)", tag)

    # ── Per-track detail (DEBUG) ───────────────────────────────────────
    for tr in tracks:
        logger.debug(
            "  %s: %.1f° (%d pts, %.1fs, score=%.4f, std=%.0f°, %d seg)",
            tr["id"], tr["mean_azimuth"], tr["n_points"],
            tr["total_duration_s"], tr["mean_score"],
            tr.get("angular_std_deg", 0), len(tr["active_segments"]),
        )

    # ── Step 6: example validation ────────────────────────────────────
    if tag == "example" and tracks:
        _validate_example_tracks(tracks)

    # ── Save output ───────────────────────────────────────────────────
    result = {
        "tag": tag,
        "n_frames": int(n_frames),
        "n_grid": int(n_grid),
        "hop_length": hop_length,
        "sample_rate": SAMPLE_RATE,
        "n_peak_points": int(points.shape[0]),
        "tracking_params": {
            "method": "global_direction_assignment",
            "n_global_directions": n_global,
            "n_windowed_directions": n_windowed,
            "n_total_directions": len(directions),
            "use_windowed_discovery": use_windowed,
            "split_broad_tracks": do_split,
            "max_assign_dist_deg": assign_dist,
            "second_peak_ratio": second_ratio,
            "min_track_points_frac": min_pt_frac,
            "min_track_duration_s": min_dur,
            "min_track_score_ratio": min_sc_ratio,
            "reject_diffuse_std_deg": max_ang_std,
        },
        "candidates": tracks,
    }

    tag_path = DOA_DIR / f"{tag}_doa_tracks.json"
    with open(tag_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    if tag == "mixture":
        canonical_path = DOA_DIR / "doa_tracks.json"
        with open(canonical_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    logger.info("[%s][track] saved -> %s", tag, tag_path)


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--tag", default="mixture",
                    help="Input tag (default: mixture)")
    main(_p.parse_args().tag)
