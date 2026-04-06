#!/usr/bin/env python3
"""
01_build_masks_from_tracks.py – DoA-guided time-frequency masks.

Reads DoA tracks (from Member 1) and the multi-channel STFT, then builds
a binary or soft TF mask for each candidate talker.

TODO:
    - Implement DoA-to-mask mapping (e.g., angular proximity weighting).
    - Support binary and soft (ratio) mask types.
    - Handle overlapping candidates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.common.json_schema import load_json
from src.common.logging_utils import setup_logging
from src.common.paths import DOA_DIR, INTERMEDIATE_DIR, ensure_output_dirs

logger = setup_logging(__name__)


def build_mask_for_candidate(
    stft: np.ndarray,
    doa_track: List[List[float]],
    n_grid: int = 360,
) -> np.ndarray:
    """
    Build a time-frequency mask for one candidate from its DoA track.

    Parameters
    ----------
    stft : np.ndarray
        Shape ``(n_channels, n_freq, n_frames)``.
    doa_track : list[list[float]]
        List of ``[frame_idx, azimuth_deg]``.
    n_grid : int
        Azimuth grid used during DoA estimation.

    Returns
    -------
    mask : np.ndarray
        Shape ``(n_freq, n_frames)`` – values in [0, 1].

    TODO
    ----
    - Implement actual mask construction from DoA track.
    - Consider beam-width and spatial aliasing.
    """
    # TODO: Implement mask building
    logger.warning("build_mask_for_candidate() is a placeholder – returning ones mask.")
    n_freq, n_frames = stft.shape[1], stft.shape[2]
    return np.ones((n_freq, n_frames), dtype=np.float32)


def main() -> None:
    """Entry point for step 01 (mask building)."""
    ensure_output_dirs()

    # Load DoA tracks
    tracks_path = DOA_DIR / "doa_tracks.json"
    if tracks_path.exists():
        scene = load_json(tracks_path)
        candidates = scene.get("candidates", [])
        logger.info("Loaded %d candidate(s) from %s", len(candidates), tracks_path)
    else:
        logger.warning("DoA tracks not found – using empty candidate list.")
        candidates = []

    # Load STFT (prefer WPE-processed if available)
    wpe_path = INTERMEDIATE_DIR / "mixture_stft_wpe.npy"
    raw_path = INTERMEDIATE_DIR / "mixture_stft.npy"
    stft_path = wpe_path if wpe_path.exists() else raw_path

    if stft_path.exists():
        stft = np.load(str(stft_path))
        logger.info("Loaded STFT: %s from %s", stft.shape, stft_path.name)
    else:
        logger.warning("No STFT found – creating dummy.")
        stft = np.zeros((4, 513, 100), dtype=np.complex64)

    # Build and save masks
    for cand in candidates:
        cid = cand.get("id", "spk00")
        track = cand.get("doa_track", [])
        mask = build_mask_for_candidate(stft, track)
        mask_path = INTERMEDIATE_DIR / f"{cid}_mask.npy"
        np.save(str(mask_path), mask)
        logger.info("Mask for %s saved → %s  shape=%s", cid, mask_path, mask.shape)

    logger.info("Step 01 (masks) complete.")


if __name__ == "__main__":
    main()
