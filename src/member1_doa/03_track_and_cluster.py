#!/usr/bin/env python3
"""
03_track_and_cluster.py – Per-frame peak extraction and DBSCAN tracking.
=========================================================================
Fourth (final) step of the Member 1 (DoA) pipeline.

What it does
------------
Takes the angular power map P(θ, t) from step 02 and extracts discrete
speaker candidates with start/end times and a smoothed DoA track.

Algorithm
---------
1.  **Per-frame peak finding**: in each time frame, pick the top-K peaks
    (default K=2) with at least ``min_peak_distance_deg`` separation.
    This yields a set of (frame_idx, azimuth_deg, score) points.

2.  **DBSCAN clustering** in (time, angle) space.  Because the angle axis
    wraps at 360°, we embed each point as (time, cos θ, sin θ) and use
    Euclidean distance with appropriately scaled axes.

    ⚠ Memory note: the previous DBSCAN attempt on raw (N×N) distance
    matrices caused OOM (~27 k × 27 k × 8 B ≈ 5.7 GB).  We avoid
    this by either:
      a. Sub-sampling if points > threshold, or
      b. Using sklearn's ball-tree–based DBSCAN which avoids the full
         matrix when ``metric='euclidean'``.

3.  **Track smoothing**: for each cluster, compute the per-frame median
    azimuth, then Gaussian-smooth with σ = ``smooth_track_sigma`` frames.

4.  **Active-segment extraction**: find contiguous blocks where the
    cluster is active and record start_sec / end_sec.

5.  **Summary**: JSON list of candidates, each with id, mean azimuth,
    active segments, and the smoothed track.

STFT convention: X[ch, f, t]  (same as earlier steps).

Outputs
-------
- ``outputs/doa/{tag}_doa_tracks.json``  — per-tag tracks
- ``outputs/doa/doa_tracks.json``        — canonical (for mixture)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.common.config import CFG, get_stft_params
from src.common.constants import SAMPLE_RATE
from src.common.logging_utils import setup_logging
from src.common.paths import DOA_DIR, ensure_output_dirs

logger = setup_logging(__name__)


# ── 1. Peak extraction ────────────────────────────────────────────────

def extract_peaks_per_frame(
    P: np.ndarray,
    top_k: int = 2,
    min_distance_deg: int = 40,
    rel_threshold: float = 0.15,
) -> np.ndarray:
    """
    Extract the top-K peaks from each frame of the angular power map.

    Parameters
    ----------
    P : np.ndarray, shape (n_grid, n_frames)
        Normalised angular power (values in [0, 1]).
    top_k : int
        Number of peaks to extract per frame.
    min_distance_deg : int
        Minimum angular distance between peaks (in grid bins, which
        equals degrees when n_grid == 360).
    rel_threshold : float
        Peaks must exceed this fraction of the frame's maximum.

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
            continue  # silent frame

        height_thr = rel_threshold * frame_max

        # ── Circular peak detection ──────────────────────────────────
        # Tile the spectrum to handle 0°/360° wrap-around.
        # We prepend and append `pad` copies so peaks near 0° or 360°
        # are detected correctly.
        pad = min_distance_deg + 2
        tiled = np.concatenate([spectrum[-pad:], spectrum, spectrum[:pad]])
        peaks, props = find_peaks(
            tiled, height=height_thr, distance=min_distance_deg,
        )
        # Map peak indices back to the original range
        peaks_orig = peaks - pad
        valid_mask = (peaks_orig >= 0) & (peaks_orig < n_grid)
        peaks_orig = peaks_orig[valid_mask]
        heights = props["peak_heights"][valid_mask]

        if len(peaks_orig) == 0:
            continue

        # Keep top-K by height
        order = np.argsort(heights)[::-1][:top_k]
        for idx in order:
            az = float(peaks_orig[idx])  # degree (0-indexed grid bin)
            sc = float(heights[idx])
            all_points.append((t, az, sc))

    if not all_points:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(all_points, dtype=np.float64)


# ── 2. Global peak detection + per-frame assignment ───────────────────

def _angular_distance(a: float, b: float) -> float:
    """Shortest angular distance in degrees (0–180)."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def find_global_directions(
    P: np.ndarray,
    min_distance_deg: int = 40,
    rel_threshold: float = 0.10,
    max_speakers: int = 8,
) -> np.ndarray:
    """
    Find dominant speaker directions from the **time-averaged** angular
    power spectrum.

    Parameters
    ----------
    P : np.ndarray, shape (n_grid, n_frames)
    min_distance_deg : int
    rel_threshold : float
    max_speakers : int

    Returns
    -------
    directions : np.ndarray, shape (K,)
        Dominant azimuths in degrees (sorted).
    """
    n_grid = P.shape[0]
    avg = P.mean(axis=1)  # (n_grid,)
    avg_max = avg.max()
    if avg_max < 1e-12:
        return np.array([], dtype=np.float64)

    height_thr = rel_threshold * avg_max

    # Circular peak detection on the average spectrum
    pad = min_distance_deg + 2
    tiled = np.concatenate([avg[-pad:], avg, avg[:pad]])
    peaks, props = find_peaks(tiled, height=height_thr,
                              distance=min_distance_deg)
    peaks_orig = peaks - pad
    valid = (peaks_orig >= 0) & (peaks_orig < n_grid)
    peaks_orig = peaks_orig[valid]
    heights = props["peak_heights"][valid]

    if len(peaks_orig) == 0:
        return np.array([], dtype=np.float64)

    # Keep top-K by height
    order = np.argsort(heights)[::-1][:max_speakers]
    dirs = np.sort(peaks_orig[order].astype(np.float64))
    return dirs


def assign_points_to_directions(
    points: np.ndarray,
    directions: np.ndarray,
    max_distance_deg: float = 20.0,
) -> np.ndarray:
    """
    Assign each per-frame peak to the nearest global direction.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        ``(frame_idx, azimuth_deg, score)``
    directions : np.ndarray, shape (K,)
        Global speaker directions in degrees.
    max_distance_deg : float
        Points farther than this from any direction are labelled -1.

    Returns
    -------
    labels : np.ndarray, shape (N,)
        Index into ``directions`` for each point, or -1 for noise.
    """
    N = points.shape[0]
    K = len(directions)
    labels = -np.ones(N, dtype=int)

    for i in range(N):
        az = points[i, 1]
        best_dist = max_distance_deg + 1
        best_k = -1
        for k in range(K):
            d = _angular_distance(az, directions[k])
            if d < best_dist:
                best_dist = d
                best_k = k
        if best_dist <= max_distance_deg:
            labels[i] = best_k

    return labels


# ── 3. Track extraction and smoothing ─────────────────────────────────

def _circular_mean(angles_deg: np.ndarray) -> float:
    """Compute the circular mean of angles in degrees."""
    rad = np.deg2rad(angles_deg)
    mean_sin = np.mean(np.sin(rad))
    mean_cos = np.mean(np.cos(rad))
    mean_rad = np.arctan2(mean_sin, mean_cos)
    return float(np.degrees(mean_rad)) % 360.0


def extract_tracks(
    points: np.ndarray,
    labels: np.ndarray,
    n_frames: int,
    hop_length: int,
    smooth_sigma: int = 5,
) -> List[Dict[str, Any]]:
    """
    Build one track per valid cluster: smoothed azimuth, active segments.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        ``(frame_idx, azimuth_deg, score)``
    labels : np.ndarray, shape (N,)
    n_frames : int
        Total number of STFT frames.
    hop_length : int
        STFT hop in samples (for time conversion).
    smooth_sigma : int
        Gaussian σ in frames for track smoothing.

    Returns
    -------
    tracks : list of dict
        Each dict follows the shared ``Candidate`` schema:
          - ``id``              : str like ``"spk00"``
          - ``doa_track``        : [[frame_idx, azimuth_deg], ...]
          - ``active_segments``  : [[start_s, end_s], ...]
          - ``mean_azimuth``     : float (circular mean, degrees)
          - ``azimuth_track``    : per-frame smoothed azimuth (or None)
          - ``n_points``         : int (raw peak count)
    """
    sr = SAMPLE_RATE
    unique_labels = sorted(set(labels) - {-1})
    tracks: List[Dict[str, Any]] = []

    for k_idx, k in enumerate(unique_labels):
        mask = labels == k
        pk = points[mask]
        frames = pk[:, 0].astype(int)
        azimuths = pk[:, 1]
        scores = pk[:, 2]

        # Circular mean azimuth
        mean_az = _circular_mean(azimuths)

        # Per-frame median azimuth (using circular median approximation)
        # We compute the circular mean per frame, then smooth.
        az_per_frame = np.full(n_frames, np.nan)
        for f in np.unique(frames):
            frame_mask = frames == f
            az_per_frame[f] = _circular_mean(azimuths[frame_mask])

        # Smooth the track (fill gaps first with interpolation)
        valid = ~np.isnan(az_per_frame)
        if valid.sum() < 2:
            continue

        # Unwrap for smoothing (avoid 0/360 jumps)
        az_unwrapped = np.copy(az_per_frame)
        valid_indices = np.where(valid)[0]
        valid_vals = az_per_frame[valid_indices]

        # Linear interpolation to fill gaps
        all_indices = np.arange(n_frames)
        # Unwrap the valid values
        rad_valid = np.deg2rad(valid_vals)
        unwrapped_rad = np.unwrap(rad_valid)
        # Interpolate
        interp_rad = np.interp(all_indices, valid_indices, unwrapped_rad)
        # Smooth
        smoothed_rad = gaussian_filter1d(interp_rad, sigma=smooth_sigma)
        smoothed_deg = np.degrees(smoothed_rad) % 360.0

        # Mark only frames where the cluster was actually active
        # (within the range of observed frames)
        f_min, f_max = int(frames.min()), int(frames.max())
        track_az = np.full(n_frames, np.nan)
        track_az[f_min:f_max + 1] = smoothed_deg[f_min:f_max + 1]

        # Active segments: contiguous blocks of observed frames
        # (merge gaps smaller than 20 frames)
        active_frames = np.sort(np.unique(frames))
        segments: List[List[float]] = []
        if len(active_frames) > 0:
            seg_start = active_frames[0]
            prev = active_frames[0]
            merge_gap = 20  # frames
            for fi in active_frames[1:]:
                if fi - prev > merge_gap:
                    # Close segment → [start_sec, end_sec]
                    t_start = float(seg_start * hop_length / sr)
                    t_end = float(prev * hop_length / sr)
                    if t_end - t_start >= 0.1:
                        segments.append([round(t_start, 3), round(t_end, 3)])
                    seg_start = fi
                prev = fi
            # Close last segment
            t_start = float(seg_start * hop_length / sr)
            t_end = float(prev * hop_length / sr)
            if t_end - t_start >= 0.1:
                segments.append([round(t_start, 3), round(t_end, 3)])

        if not segments:
            continue

        # Build doa_track: [[frame_idx, azimuth_deg], ...] for frames
        # where the speaker was observed (matches Candidate schema).
        doa_track: List[List[float]] = []
        for f_idx in range(n_frames):
            if not np.isnan(track_az[f_idx]):
                doa_track.append([f_idx, round(float(track_az[f_idx]), 1)])

        tracks.append({
            "id": f"spk{k_idx:02d}",
            "mean_azimuth": round(mean_az, 1),
            "n_points": int(mask.sum()),
            "doa_track": doa_track,
            "active_segments": segments,
            "azimuth_track": [
                round(float(v), 1) if not np.isnan(v) else None
                for v in track_az
            ],
        })

    # Sort by mean azimuth for consistent ordering
    tracks.sort(key=lambda t: t["mean_azimuth"])
    # Re-number ids sequentially as "spk00", "spk01", …
    for i, tr in enumerate(tracks):
        tr["id"] = f"spk{i:02d}"

    return tracks


# ── Entry point ────────────────────────────────────────────────────────

def main(tag: str = "mixture") -> None:
    """
    Run tracking / clustering for the given tag.

    Parameters
    ----------
    tag : str
        Input tag, e.g. ``"example"`` or ``"mixture"``.
    """
    ensure_output_dirs()

    doa_cfg = CFG.get("doa", {})
    top_k = doa_cfg.get("top_k_peaks_per_frame", 2)
    min_dist = doa_cfg.get("min_peak_distance_deg", 40)
    rel_thr = doa_cfg.get("peak_rel_threshold", 0.15)
    eps_deg = doa_cfg.get("dbscan_eps_deg", 15.0)
    min_samples = doa_cfg.get("dbscan_min_samples", 30)
    smooth_sigma = doa_cfg.get("smooth_track_sigma", 5)

    stft_params = get_stft_params()
    hop_length = stft_params.get("hop_length", 256)

    # Load angular power map from step 02
    post_path = DOA_DIR / f"{tag}_doa_posteriors.npy"
    if not post_path.exists():
        raise FileNotFoundError(
            f"DoA posteriors not found: {post_path}  — run step 02 first."
        )

    P = np.load(str(post_path))
    n_grid, n_frames = P.shape
    logger.info("[%s] Loaded posteriors: %s", tag, P.shape)

    # 1. Extract peaks
    logger.info("[%s] Extracting top-%d peaks per frame (min_dist=%d°) …",
                tag, top_k, min_dist)
    points = extract_peaks_per_frame(P, top_k=top_k,
                                     min_distance_deg=min_dist,
                                     rel_threshold=rel_thr)
    logger.info("[%s] Extracted %d peak points.", tag, points.shape[0])

    if points.shape[0] == 0:
        logger.warning("[%s] No peaks found – writing empty tracks.", tag)
        tracks: List[Dict[str, Any]] = []
    else:
        # 2. Find global speaker directions from time-averaged spectrum
        logger.info("[%s] Finding global directions (min_dist=%d°) …",
                    tag, min_dist)
        directions = find_global_directions(
            P, min_distance_deg=min_dist, rel_threshold=rel_thr,
        )
        logger.info("[%s] Global directions: %s",
                    tag, [f"{d:.0f}°" for d in directions])

        # 3. Assign per-frame peaks to nearest global direction
        max_assign_dist = eps_deg  # reuse DBSCAN eps as assignment radius
        labels = assign_points_to_directions(
            points, directions, max_distance_deg=max_assign_dist,
        )
        n_assigned = int((labels >= 0).sum())
        n_noise = int((labels == -1).sum())
        logger.info("[%s] Assigned %d points to %d directions, %d noise.",
                    tag, n_assigned, len(directions), n_noise)

        # 4. Extract smoothed tracks
        tracks = extract_tracks(points, labels, n_frames,
                                hop_length=hop_length,
                                smooth_sigma=smooth_sigma)

    logger.info("[%s] Extracted %d speaker tracks:", tag, len(tracks))
    for tr in tracks:
        logger.info(
            "  %s: %.1f° (%d points, %d segment(s))",
            tr["id"], tr["mean_azimuth"], tr["n_points"],
            len(tr["active_segments"]),
        )

    # Prepare output dict
    result = {
        "tag": tag,
        "n_frames": int(n_frames),
        "n_grid": int(n_grid),
        "hop_length": hop_length,
        "sample_rate": SAMPLE_RATE,
        "n_peak_points": int(points.shape[0]),
        "clustering_params": {
            "method": "global_peak_assignment",
            "assignment_radius_deg": eps_deg,
            "min_peak_distance_deg": min_dist,
        },
        "candidates": tracks,
    }

    # Save per-tag
    tag_path = DOA_DIR / f"{tag}_doa_tracks.json"
    with open(tag_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info("[%s] Saved → %s", tag, tag_path)

    # Canonical copy for "mixture"
    if tag == "mixture":
        canonical_path = DOA_DIR / "doa_tracks.json"
        with open(canonical_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        logger.info("[%s] Saved canonical → %s", tag, canonical_path)

    logger.info("Step 03 [%s] complete.", tag)


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--tag", default="mixture",
                    help="Input tag (default: mixture)")
    main(_p.parse_args().tag)
