#!/usr/bin/env python3
"""
02_doa_estimate.py – Direction-of-arrival estimation via GCC-PHAT.

Reads the multi-channel STFT and calibration data, then builds a
time–azimuth DoA posterior (heatmap) using frame-wise GCC-PHAT TDOA
estimates mapped to azimuth via a known microphone geometry.

The result is saved as a NumPy array to ``outputs/doa/doa_posteriors.npy``.

Algorithm
---------
For each STFT frame:
  1. For each mic pair, compute the normalised cross-correlation using
     PHAT weighting (in the frequency domain, per-frame).
  2. Convert each candidate TDOA to an azimuth angle using the
     microphone geometry (inter-mic distance projected onto each
     look direction).
  3. Accumulate a steered-response-power (SRP-PHAT) pseudo-spectrum
     by summing the real part of the PHAT-weighted cross-spectrum
     steered to each candidate azimuth.
  4. Normalise each frame's spectrum to sum to 1 (pseudo-probability).

Microphone geometry (BTE hearing-aid pair, horizontal plane):
    LF = (+0.006, +0.0875) m      (Left-Front)
    LR = (-0.006, +0.0875) m      (Left-Rear)
    RF = (+0.006, -0.0875) m      (Right-Front)
    RR = (-0.006, -0.0875) m      (Right-Rear)

    x-axis → forward,  y-axis → left
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.common.config import CFG, get_stft_params
from src.common.constants import CHANNEL_ORDER, SAMPLE_RATE
from src.common.logging_utils import setup_logging
from src.common.paths import CALIB_DIR, DOA_DIR, INTERMEDIATE_DIR, ensure_output_dirs

logger = setup_logging(__name__)

# ── Microphone geometry (metres) ───────────────────────────────────────
# BTE hearing-aid pair on head; x = forward, y = left
MIC_POSITIONS: Dict[str, np.ndarray] = {
    "LF": np.array([+0.006, +0.0875]),
    "LR": np.array([-0.006, +0.0875]),
    "RF": np.array([+0.006, -0.0875]),
    "RR": np.array([-0.006, -0.0875]),
}

SPEED_OF_SOUND: float = 343.0  # m/s

# All 6 mic pairs (same order as calibration)
MIC_PAIRS: List[Tuple[str, str]] = [
    ("LF", "LR"), ("LF", "RF"), ("LF", "RR"),
    ("LR", "RF"), ("LR", "RR"), ("RF", "RR"),
]


def load_calibration(path: Path) -> Dict:
    """Load calibration JSON produced by step 01."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _precompute_steering_delays(
    n_grid: int,
    sr: int,
    freq_bins: np.ndarray,
) -> np.ndarray:
    """
    Precompute the expected inter-mic delay (in radians of phase shift)
    for every (pair, azimuth, frequency) combination.

    Returns
    -------
    steering : np.ndarray
        Shape ``(n_pairs, n_grid, n_freq)`` – complex phase shift
        e^{-j 2π f τ} for each pair/azimuth/freq.
    """
    n_pairs = len(MIC_PAIRS)
    n_freq = len(freq_bins)
    azimuths = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)  # radians

    # Unit direction vectors for each azimuth (x = forward, y = left)
    # azimuth = 0 → forward, 90° → left, 180° → behind, 270° → right
    directions = np.stack([np.cos(azimuths), np.sin(azimuths)], axis=-1)  # (n_grid, 2)

    steering = np.empty((n_pairs, n_grid, n_freq), dtype=np.complex128)
    for p_idx, (m1, m2) in enumerate(MIC_PAIRS):
        d_vec = MIC_POSITIONS[m1] - MIC_POSITIONS[m2]  # (2,)
        # Projected delay for each azimuth (seconds)
        tau = directions @ d_vec / SPEED_OF_SOUND       # (n_grid,)
        # Phase shift: e^{-j 2π f τ}
        phase = -2.0 * np.pi * np.outer(tau, freq_bins)  # (n_grid, n_freq)
        steering[p_idx] = np.exp(1j * phase)

    return steering


def estimate_doa_heatmap(
    stft: np.ndarray,
    calibration: Dict,
    n_grid: int = 360,
    freq_range: Tuple[int, int] = (300, 8000),
) -> np.ndarray:
    """
    Estimate a time–azimuth DoA posterior using SRP-PHAT.

    Parameters
    ----------
    stft : np.ndarray
        Complex STFT, shape ``(n_channels, n_freq, n_frames)``.
    calibration : dict
        Calibration data (unused for now; geometry-based steering used).
    n_grid : int
        Number of azimuth bins (0°–360°).
    freq_range : tuple[int, int]
        Frequency range used for DoA estimation (Hz).

    Returns
    -------
    heatmap : np.ndarray
        Shape ``(n_frames, n_grid)`` – pseudo-probability for each
        (frame, azimuth) cell.
    """
    params = get_stft_params()
    n_fft = params["n_fft"]
    sr = SAMPLE_RATE
    n_channels, n_freq, n_frames = stft.shape

    # Frequency axis
    freq_bins = np.linspace(0, sr / 2, n_freq)

    # Restrict to useful frequency range
    f_mask = (freq_bins >= freq_range[0]) & (freq_bins <= freq_range[1])
    freq_sub = freq_bins[f_mask]
    n_fsub = int(f_mask.sum())

    logger.info(
        "SRP-PHAT: %d azimuth bins, freq %d–%d Hz (%d bins), %d frames",
        n_grid, freq_range[0], freq_range[1], n_fsub, n_frames,
    )

    # Precompute steering vectors for the sub-band
    steering = _precompute_steering_delays(n_grid, sr, freq_sub)
    # steering shape: (n_pairs, n_grid, n_fsub)

    # Map channel names to indices
    ch_idx = {name: i for i, name in enumerate(CHANNEL_ORDER)}

    heatmap = np.zeros((n_frames, n_grid), dtype=np.float64)

    # For efficiency, extract sub-band STFT once
    stft_sub = stft[:, f_mask, :]  # (n_channels, n_fsub, n_frames)

    # SRP-PHAT accumulation
    for p_idx, (m1, m2) in enumerate(MIC_PAIRS):
        i1, i2 = ch_idx[m1], ch_idx[m2]

        # Cross-spectrum for this pair: (n_fsub, n_frames)
        X12 = stft_sub[i1] * np.conj(stft_sub[i2])

        # PHAT weighting
        mag = np.abs(X12) + 1e-12
        X12_phat = X12 / mag  # (n_fsub, n_frames)

        # Steer and accumulate: for each azimuth θ and frame t,
        #   P(θ, t) += Re{ Σ_f  steering(p, θ, f) * X12_phat(f, t) }
        # steering[p_idx] is (n_grid, n_fsub), X12_phat is (n_fsub, n_frames)
        # Result per pair: (n_grid, n_frames)
        contribution = np.real(steering[p_idx] @ X12_phat)  # (n_grid, n_frames)
        heatmap += contribution.T  # (n_frames, n_grid)

    # Normalise each frame to [0, 1] range then to pseudo-probability
    for t in range(n_frames):
        row = heatmap[t]
        row_min = row.min()
        row -= row_min
        row_sum = row.sum()
        if row_sum > 1e-12:
            row /= row_sum
        else:
            row[:] = 1.0 / n_grid

    return heatmap.astype(np.float32)


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
