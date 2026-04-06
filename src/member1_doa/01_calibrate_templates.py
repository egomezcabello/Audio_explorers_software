#!/usr/bin/env python3
"""
01_calibrate_templates.py – Array calibration via template measurements.

This script creates or loads calibration data (e.g., inter-microphone
time-delay templates measured from a known source position) and stores the
result to ``outputs/calib/calibration.json``.

Channel order (always):
    ["LF", "LR", "RF", "RR"]

TODO:
    - Implement GCC-PHAT cross-correlation between mic pairs.
    - Compute time-delay-of-arrival (TDOA) templates for known angles.
    - Optionally load pre-computed calibration from file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

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


# ── Placeholder functions ──────────────────────────────────────────────
def gcc_phat(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sr: int = SAMPLE_RATE,
    max_delay_samples: int = 64,
) -> tuple[float, float]:
    """
    Compute the GCC-PHAT cross-correlation between two signals and return
    the estimated time-delay-of-arrival (TDOA).

    Parameters
    ----------
    sig1, sig2 : np.ndarray
        Single-channel time-domain signals.
    sr : int
        Sample rate.
    max_delay_samples : int
        Maximum lag to search, in samples.

    Returns
    -------
    tdoa_samples : float
        Delay in samples (positive → sig2 arrives later).
    peak_value : float
        Height of the GCC-PHAT peak (0–1 range, higher = more confident).

    TODO
    ----
    - Implement the actual GCC-PHAT algorithm.
    - Consider frequency-domain weighting and interpolation.
    """
    # TODO: Implement GCC-PHAT
    logger.warning("gcc_phat() is a placeholder – returning zeros.")
    return 0.0, 0.0


def calibrate_from_stft(
    stft_path: Path,
) -> CalibrationBundle:
    """
    Run pair-wise calibration on a multi-channel STFT.

    Parameters
    ----------
    stft_path : Path
        Path to a ``.npy`` file with shape ``(n_channels, n_freq, n_frames)``.

    Returns
    -------
    CalibrationBundle

    TODO
    ----
    - Load the STFT and convert selected pairs back to time domain.
    - Call gcc_phat for each pair.
    """
    # TODO: Implement calibration from STFT
    logger.warning("calibrate_from_stft() is a placeholder.")

    bundle = CalibrationBundle()
    mic_pairs = [("LF", "LR"), ("LF", "RF"), ("LF", "RR"),
                 ("LR", "RF"), ("LR", "RR"), ("RF", "RR")]
    for m1, m2 in mic_pairs:
        bundle.pairs.append(
            CalibrationResult(mic_pair=[m1, m2], tdoa_samples=0.0,
                              tdoa_seconds=0.0, confidence=0.0)
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
    logger.info("Step 01 complete.")


if __name__ == "__main__":
    main()
