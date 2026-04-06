#!/usr/bin/env python3
"""
03_track_and_cluster.py – Track and cluster DoA estimates into candidates.

Reads the DoA posterior heatmap, finds persistent peaks, and clusters them
into candidate talker objects.  Results are saved as
``outputs/doa/doa_tracks.json``.

TODO:
    - Implement peak-picking on the azimuth posterior per frame.
    - Implement temporal tracking (e.g., Kalman filter or simple smoothing).
    - Cluster nearby tracks into unique talker candidates.
    - Write candidate list in the shared JSON schema.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np

from src.common.json_schema import Candidate, SceneSummary, save_json
from src.common.logging_utils import setup_logging
from src.common.paths import DOA_DIR, ensure_output_dirs

logger = setup_logging(__name__)


def pick_peaks(
    heatmap: np.ndarray,
    threshold: float = 0.5,
) -> List[List[float]]:
    """
    Find azimuth peaks in each frame of the DoA posterior.

    Parameters
    ----------
    heatmap : np.ndarray
        Shape ``(n_frames, n_grid)``.
    threshold : float
        Minimum normalised energy to count as a peak.

    Returns
    -------
    peaks : list[list[float]]
        List of ``[frame_idx, azimuth_deg]`` pairs.

    TODO
    ----
    - Implement actual peak-picking (scipy.signal.find_peaks or similar).
    """
    # TODO: Implement peak picking
    logger.warning("pick_peaks() is a placeholder – returning empty list.")
    return []


def cluster_tracks(
    peaks: List[List[float]],
    min_track_length: int = 10,
) -> List[Candidate]:
    """
    Cluster detected peaks into persistent talker candidates.

    Parameters
    ----------
    peaks : list[list[float]]
        Output of ``pick_peaks()``.
    min_track_length : int
        Minimum number of frames for a valid track.

    Returns
    -------
    candidates : list[Candidate]

    TODO
    ----
    - Implement clustering (e.g., DBSCAN on azimuth trajectory).
    - Assign speaker IDs as spk00, spk01, …
    """
    # TODO: Implement clustering
    logger.warning("cluster_tracks() is a placeholder – returning one dummy candidate.")
    dummy = Candidate(
        id="spk00",
        doa_track=[],
        active_segments=[],
    )
    return [dummy]


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

    peaks = pick_peaks(heatmap)
    candidates = cluster_tracks(peaks)

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
