#!/usr/bin/env python3
"""
01_calibrate_templates.py – Array calibration via GCC-PHAT.

Computes pairwise inter-microphone time-delay-of-arrival (TDOA)
estimates for all six mic pairs from the multi-channel mixture STFT.
Results are saved to ``outputs/calib/calibration.json``.

Algorithm
---------
For each microphone pair (i, j):
  1. Convert selected STFT channels back to time domain via iSTFT.
  2. Compute the Generalised Cross-Correlation with PHase Transform
     (GCC-PHAT) in the frequency domain.
  3. Find the peak of the cross-correlation within a plausible lag
     range (±max_delay_samples) and apply parabolic interpolation
     for sub-sample accuracy.

Channel order (always):
    ["LF", "LR", "RF", "RR"]
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import istft as _istft

from src.common.config import get_stft_params
from src.common.constants import CHANNEL_ORDER, SAMPLE_RATE
from src.common.logging_utils import setup_logging
from src.common.paths import CALIB_DIR, INTERMEDIATE_DIR, ensure_output_dirs

logger = setup_logging(__name__)


# ── Data structures ────────────────────────────────────────────────────
@dataclass
class CalibrationResult:
    """Holds the calibration information for one microphone pair."""

    mic_pair: List[str] = field(default_factory=lambda: ["LF", "LR"])
    tdoa_samples: float = 0.0  # estimated delay in samples
    tdoa_seconds: float = 0.0  # estimated delay in seconds
    confidence: float = 0.0    # GCC-PHAT peak height


@dataclass
class CalibrationBundle:
    """Collection of calibration results for all mic pairs."""

    sample_rate: int = SAMPLE_RATE
    channel_order: List[str] = field(default_factory=lambda: list(CHANNEL_ORDER))
    pairs: List[CalibrationResult] = field(default_factory=list)


# ── GCC-PHAT implementation ───────────────────────────────────────────
def gcc_phat(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sr: int = SAMPLE_RATE,
    max_delay_samples: int = 64,
) -> tuple[float, float]:
    """
    Compute GCC-PHAT cross-correlation and return the TDOA.

    Parameters
    ----------
    sig1, sig2 : np.ndarray  (1-D, float)
        Single-channel time-domain signals (same length).
    sr : int
        Sample rate.
    max_delay_samples : int
        Maximum lag to search, in samples.

    Returns
    -------
    tdoa_samples : float
        Delay in samples (positive ⇒ sig2 arrives later).
        Sub-sample resolution via parabolic interpolation.
    peak_value : float
        Height of the GCC-PHAT peak (≥ 0, higher = more confident).
    """
    n = len(sig1) + len(sig2) - 1          # full linear correlation length
    n_fft = 1 << int(np.ceil(np.log2(n)))  # next power of 2

    S1 = np.fft.rfft(sig1, n=n_fft)
    S2 = np.fft.rfft(sig2, n=n_fft)

    # Cross-power spectrum with PHAT weighting
    G = S1 * np.conj(S2)
    denom = np.abs(G) + 1e-12              # avoid division by zero
    G_phat = G / denom

    # Back to time domain
    cc = np.fft.irfft(G_phat, n=n_fft)

    # Restrict search to ±max_delay_samples
    max_d = min(max_delay_samples, n_fft // 2 - 1)
    lags = np.concatenate([cc[-max_d:], cc[: max_d + 1]])
    lag_indices = np.arange(-max_d, max_d + 1)

    peak_idx = int(np.argmax(np.abs(lags)))
    peak_val = float(np.abs(lags[peak_idx]))
    coarse_delay = int(lag_indices[peak_idx])

    # Parabolic interpolation for sub-sample accuracy
    if 0 < peak_idx < len(lags) - 1:
        alpha = float(np.abs(lags[peak_idx - 1]))
        beta = float(np.abs(lags[peak_idx]))
        gamma = float(np.abs(lags[peak_idx + 1]))
        denom_p = alpha - 2.0 * beta + gamma
        if abs(denom_p) > 1e-12:
            p = 0.5 * (alpha - gamma) / denom_p
        else:
            p = 0.0
        tdoa = coarse_delay + p
    else:
        tdoa = float(coarse_delay)

    return tdoa, peak_val


def _istft_single_channel(
    stft_ch: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Reconstruct time-domain signal from a single-channel STFT."""
    params = get_stft_params()
    win_len = params["win_length"]
    hop = params["hop_length"]
    n_fft = params["n_fft"]
    window = params["window"]

    _, x = _istft(stft_ch, fs=sr, window=window, nperseg=win_len,
                  noverlap=win_len - hop, nfft=n_fft)
    return x.astype(np.float64)


def calibrate_from_stft(
    stft_path: Path,
    max_delay_samples: int = 64,
) -> CalibrationBundle:
    """
    Run pair-wise GCC-PHAT calibration on a multi-channel STFT.

    Parameters
    ----------
    stft_path : Path
        Path to a ``.npy`` file with shape ``(n_channels, n_freq, n_frames)``.
    max_delay_samples : int
        Maximum lag to search within (in samples).

    Returns
    -------
    CalibrationBundle
        Contains TDOA estimates for all 6 mic pairs.
    """
    if not stft_path.exists():
        logger.warning("STFT file not found: %s – returning empty calibration.", stft_path)
        return CalibrationBundle()

    stft = np.load(str(stft_path))
    logger.info("Loaded STFT: %s", stft.shape)

    # Convert each channel back to time domain
    n_ch = stft.shape[0]
    time_signals: List[np.ndarray] = []
    for ch in range(n_ch):
        sig = _istft_single_channel(stft[ch])
        time_signals.append(sig)
        logger.info("  iSTFT ch %d (%s): %d samples", ch, CHANNEL_ORDER[ch], len(sig))

    # All 6 mic pairs
    mic_pairs = [("LF", "LR"), ("LF", "RF"), ("LF", "RR"),
                 ("LR", "RF"), ("LR", "RR"), ("RF", "RR")]

    bundle = CalibrationBundle()
    for m1, m2 in mic_pairs:
        idx1 = CHANNEL_ORDER.index(m1)
        idx2 = CHANNEL_ORDER.index(m2)
        tdoa_samp, conf = gcc_phat(
            time_signals[idx1], time_signals[idx2],
            sr=SAMPLE_RATE, max_delay_samples=max_delay_samples,
        )
        tdoa_sec = tdoa_samp / SAMPLE_RATE
        result = CalibrationResult(
            mic_pair=[m1, m2],
            tdoa_samples=round(tdoa_samp, 4),
            tdoa_seconds=round(tdoa_sec, 8),
            confidence=round(conf, 6),
        )
        bundle.pairs.append(result)
        logger.info(
            "  Pair (%s, %s): TDOA = %.2f samples (%.4f ms), conf = %.4f",
            m1, m2, tdoa_samp, tdoa_sec * 1000, conf,
        )

    return bundle


def save_calibration(bundle: CalibrationBundle, path: Path) -> None:
    """Serialise calibration results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(bundle), fh, indent=2)
    logger.info("Calibration saved → %s", path)


def main() -> None:
    """Entry point for step 01."""
    ensure_output_dirs()

    stft_path = INTERMEDIATE_DIR / "mixture_stft.npy"
    bundle = calibrate_from_stft(stft_path)
    save_calibration(bundle, CALIB_DIR / "calibration.json")
    logger.info("Step 01 complete – %d pair(s) calibrated.", len(bundle.pairs))


if __name__ == "__main__":
    main()
