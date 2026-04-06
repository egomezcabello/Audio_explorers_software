#!/usr/bin/env python3
"""
02_doa_estimate.py – Direction-of-arrival estimation.

Reads the multi-channel STFT and calibration data, then estimates a
time–azimuth DoA posterior (heatmap).  The result is saved as a NumPy
array to ``outputs/doa/doa_posteriors.npy``.

Channel order (always):
    ["LF", "LR", "RF", "RR"]

TODO:
    - Implement SRP-PHAT or GCC-PHAT-based DoA grid search.
    - Optionally implement MUSIC or other narrowband methods.
    - Store per-frame azimuth posteriors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.common.config import CFG
from src.common.constants import SAMPLE_RATE
from src.common.logging_utils import setup_logging
from src.common.paths import CALIB_DIR, DOA_DIR, INTERMEDIATE_DIR, ensure_output_dirs

logger = setup_logging(__name__)


def load_calibration(path: Path) -> Dict:
    """Load calibration JSON produced by step 01."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def estimate_doa_heatmap(
    stft: np.ndarray,
    calibration: Dict,
    n_grid: int = 360,
    freq_range: tuple[int, int] = (300, 8000),
) -> np.ndarray:
    """
    Estimate a time–azimuth DoA posterior from the multi-channel STFT.

    Parameters
    ----------
    stft : np.ndarray
        Complex STFT, shape ``(n_channels, n_freq, n_frames)``.
    calibration : dict
        Calibration data (mic-pair TDOAs).
    n_grid : int
        Number of azimuth bins (0°–360°).
    freq_range : tuple[int, int]
        Frequency range used for DoA estimation (Hz).

    Returns
    -------
    heatmap : np.ndarray
        Shape ``(n_frames, n_grid)`` – pseudo-probability for each
        (frame, azimuth) pair.

    TODO
    ----
    - Implement SRP-PHAT grid search.
    - Apply frequency weighting.
    - Normalise across azimuth for each frame.
    """
    # TODO: Implement actual DoA estimation
    logger.warning("estimate_doa_heatmap() is a placeholder – returning uniform.")
    n_frames = stft.shape[2]
    heatmap = np.ones((n_frames, n_grid), dtype=np.float32) / n_grid
    return heatmap


def main() -> None:
    """Entry point for step 02."""
    ensure_output_dirs()

    doa_cfg = CFG.get("doa", {})
    n_grid = doa_cfg.get("n_grid", 360)
    freq_range = tuple(doa_cfg.get("freq_range", [300, 8000]))

    stft_path = INTERMEDIATE_DIR / "mixture_stft.npy"
    calib_path = CALIB_DIR / "calibration.json"

    if stft_path.exists():
        stft = np.load(str(stft_path))
        logger.info("Loaded STFT: %s", stft.shape)
    else:
        logger.warning("STFT not found at %s – creating dummy.", stft_path)
        stft = np.zeros((4, 513, 100), dtype=np.complex64)

    if calib_path.exists():
        calibration = load_calibration(calib_path)
    else:
        logger.warning("Calibration not found – using empty dict.")
        calibration = {}

    heatmap = estimate_doa_heatmap(stft, calibration, n_grid=n_grid,
                                    freq_range=freq_range)
    out_path = DOA_DIR / "doa_posteriors.npy"
    np.save(str(out_path), heatmap)
    logger.info("DoA posteriors saved → %s  shape=%s", out_path, heatmap.shape)
    logger.info("Step 02 complete.")


if __name__ == "__main__":
    main()
