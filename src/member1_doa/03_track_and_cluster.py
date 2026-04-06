#!/usr/bin/env python3
"""
03_track_and_cluster.py – Track and cluster DoA estimates into candidates.

Reads the DoA posterior heatmap, finds persistent azimuth peaks, and
clusters them into candidate talker objects.

Results are saved as ``outputs/doa/doa_tracks.json``.

Algorithm  (memory-efficient two-stage approach)
-------------------------------------------------
1. **Global peak finding** – Average the heatmap across all frames,
   smooth circularly, and run ``scipy.signal.find_peaks`` to identify
   the dominant azimuth angles in the scene.
2. **Frame-level assignment** – For each frame, pick the best azimuth
   among the global candidates (nearest peak above a threshold).
   This produces one DoA track per candidate.
3. **Active-segment extraction** – For each candidate, find contiguous
   blocks of frames where it was the dominant source.
4. **Output** – Write a ``SceneSummary`` JSON with one ``Candidate``
   per detected talker direction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from src.common.config import CFG, get_stft_params
from src.common.constants import SAMPLE_RATE
from src.common.json_schema import Candidate, SceneSummary, save_json
from src.common.logging_utils import setup_logging
from src.common.paths import DOA_DIR, ensure_output_dirs

logger = setup_logging(__name__)


def _smooth_circular(row: np.ndarray, sigma_bins: int = 5) -> np.ndarray:
    """
    Smooth a 1-D azimuth distribution with a circular (wrap-around)
    uniform filter.
    """
    padded = np.concatenate([row[-sigma_bins:], row, row[:sigma_bins]])
    smoothed = uniform_filter1d(padded, size=2 * sigma_bins + 1)
    return smoothed[sigma_bins: sigma_bins + len(row)]


def _circular_distance_vec(a: np.ndarray, b: float) -> np.ndarray:
    """Shortest angular distance (degrees) between array *a* and scalar *b*."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def find_dominant_azimuths(
    heatmap: np.ndarray,
    smooth_sigma: int = 7,
    min_distance_deg: int = 20,
    rel_threshold: float = 0.15,
) -> np.ndarray:
    """
    Find dominant azimuth directions from the time-averaged heatmap.

    Returns
    -------
    azimuths_deg : np.ndarray
        1-D array of dominant azimuth angles (degrees).
    """
    n_grid = heatmap.shape[1]
    deg_per_bin = 360.0 / n_grid
    min_dist_bins = max(1, int(min_distance_deg / deg_per_bin))

    # Time-average and smooth
    avg = heatmap.mean(axis=0).astype(np.float64)
    avg_smooth = _smooth_circular(avg, sigma_bins=smooth_sigma)

    max_val = avg_smooth.max()
    height_thresh = rel_threshold * max_val

    peaks_idx, props = find_peaks(
        avg_smooth,
        height=height_thresh,
        distance=min_dist_bins,
    )

    azimuths = peaks_idx * deg_per_bin
    logger.info(
        "Global dominant azimuths: %s° (from %d-bin averaged heatmap)",
        [f"{a:.0f}" for a in azimuths], n_grid,
    )
    return azimuths


def build_tracks(
    heatmap: np.ndarray,
    dominant_azimuths: np.ndarray,
    hop_length: int = 256,
    sr: int = SAMPLE_RATE,
    assignment_radius_deg: float = 25.0,
    activity_threshold: float = 0.5,
) -> List[Candidate]:
    """
    Assign each frame to the closest dominant azimuth and build per-candidate
    DoA tracks.

    Parameters
    ----------
    heatmap : np.ndarray
        Shape ``(n_frames, n_grid)``.
    dominant_azimuths : np.ndarray
        Angles (degrees) from ``find_dominant_azimuths()``.
    hop_length : int
        STFT hop length for frame→time conversion.
    sr : int
        Sample rate.
    assignment_radius_deg : float
        Maximum angular distance for a frame to be assigned to a candidate.
    activity_threshold : float
        Minimum fraction of the frame's peak energy (relative to the
        frame max) for the candidate to be considered active.

    Returns
    -------
    candidates : list[Candidate]
    """
    n_frames, n_grid = heatmap.shape
    deg_per_bin = 360.0 / n_grid
    frame_to_sec = hop_length / sr
    n_cands = len(dominant_azimuths)

    if n_cands == 0:
        logger.warning("No dominant azimuths – returning empty candidate list.")
        return []

    # For each frame, look up the heatmap value at each candidate's bin
    cand_bins = np.round(dominant_azimuths / deg_per_bin).astype(int) % n_grid

    # Pre-extract the heatmap columns at candidate bins → (n_frames, n_cands)
    cand_energy = heatmap[:, cand_bins]

    # Frame-wise maximum energy
    frame_max = heatmap.max(axis=1, keepdims=True)
    frame_max = np.maximum(frame_max, 1e-12)

    # Relative energy at each candidate position
    rel_energy = cand_energy / frame_max  # (n_frames, n_cands)

    # Also compute per-frame "refined" azimuth near each candidate
    # by taking a weighted average in a ±assignment_radius_deg window
    half_win = int(np.ceil(assignment_radius_deg / deg_per_bin))

    # Build candidate tracks
    candidates: List[Candidate] = []
    for ci in range(n_cands):
        center_bin = cand_bins[ci]
        active_mask = rel_energy[:, ci] >= activity_threshold

        if active_mask.sum() < 5:
            logger.info("  Candidate at %.0f° – too few active frames (%d), skipping.",
                        dominant_azimuths[ci], active_mask.sum())
            continue

        # Refined azimuth per frame (weighted centroid in local window)
        doa_track: List[List[float]] = []
        for t in range(n_frames):
            if not active_mask[t]:
                continue
            # Extract local window with circular wrapping
            indices = np.arange(center_bin - half_win, center_bin + half_win + 1) % n_grid
            local_vals = heatmap[t, indices].astype(np.float64)
            local_angles = indices * deg_per_bin
            # Handle wrap-around for weighted average
            # Shift angles relative to centre
            center_deg = dominant_azimuths[ci]
            shifted = (local_angles - center_deg + 180) % 360 - 180
            weight_sum = local_vals.sum()
            if weight_sum > 1e-12:
                refined = center_deg + np.average(shifted, weights=local_vals)
                refined = refined % 360.0
            else:
                refined = center_deg
            doa_track.append([float(t), round(float(refined), 2)])

        if len(doa_track) < 5:
            continue

        # Active segments
        active_times = np.array([p[0] for p in doa_track]) * frame_to_sec
        segments = _find_active_segments(active_times, gap_threshold=0.5)

        mean_az = float(np.mean([p[1] for p in doa_track]))
        spk_id = f"spk{len(candidates):02d}"

        candidates.append(Candidate(
            id=spk_id,
            doa_track=doa_track,
            active_segments=segments,
        ))

        logger.info(
            "  %s: mean_az=%.1f°, %d active frames (%.1f s), %d segment(s)",
            spk_id, mean_az, len(doa_track),
            len(doa_track) * frame_to_sec, len(segments),
        )

    # Sort by mean azimuth
    candidates.sort(
        key=lambda c: np.mean([p[1] for p in c.doa_track]) if c.doa_track else 0
    )
    for i, c in enumerate(candidates):
        c.id = f"spk{i:02d}"

    return candidates


def _find_active_segments(
    sorted_times: np.ndarray,
    gap_threshold: float = 1.0,
) -> List[List[float]]:
    """
    Group sorted timestamps into contiguous segments.

    A new segment starts when the gap between consecutive detections
    exceeds *gap_threshold* seconds.
    """
    if len(sorted_times) == 0:
        return []

    segments: List[List[float]] = []
    seg_start = float(sorted_times[0])
    prev = seg_start

    for t in sorted_times[1:]:
        if t - prev > gap_threshold:
            segments.append([round(seg_start, 4), round(prev, 4)])
            seg_start = float(t)
        prev = float(t)

    segments.append([round(seg_start, 4), round(prev, 4)])
    return segments


def main() -> None:
    """Entry point for step 03."""
    ensure_output_dirs()

    posteriors_path = DOA_DIR / "doa_posteriors.npy"

    if posteriors_path.exists():
        heatmap = np.load(str(posteriors_path))
        logger.info("Loaded DoA posteriors: %s", heatmap.shape)
    else:
        logger.warning("DoA posteriors not found – using dummy array.")
        heatmap = np.ones((100, 360), dtype=np.float32) / 360

    # Stage 1: find dominant azimuth directions
    dominant_azimuths = find_dominant_azimuths(
        heatmap,
        smooth_sigma=7,
        min_distance_deg=20,
        rel_threshold=0.15,
    )

    # Stage 2: build per-candidate tracks
    stft_params = get_stft_params()
    candidates = build_tracks(
        heatmap,
        dominant_azimuths,
        hop_length=stft_params["hop_length"],
        sr=SAMPLE_RATE,
        assignment_radius_deg=25.0,
        activity_threshold=0.5,
    )

    # Build a SceneSummary and save
    summary = SceneSummary(candidates=candidates)
    out_path = DOA_DIR / "doa_tracks.json"
    save_json(summary, out_path)
    logger.info(
        "Saved %d candidate(s) → %s", len(candidates), out_path
    )
    logger.info("Step 03 complete.")


if __name__ == "__main__":
    main()
