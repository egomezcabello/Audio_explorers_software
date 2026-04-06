#!/usr/bin/env python3
"""
00_wpe_optional.py – Optional WPE dereverberation.

Applies Weighted Prediction Error (WPE) dereverberation to the
multi-channel STFT if enabled in ``config.yaml``.

Channel order (always):
    ["LF", "LR", "RF", "RR"]

TODO:
    - Integrate ``nara_wpe`` for online or offline WPE.
    - Save the dereverberated STFT to outputs/intermediate/.
    - Add a bypass flag that simply copies the input if WPE is disabled.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.common.config import CFG
from src.common.logging_utils import setup_logging
from src.common.paths import INTERMEDIATE_DIR, ensure_output_dirs

logger = setup_logging(__name__)


def apply_wpe(
    stft: np.ndarray,
    taps: int = 10,
    delay: int = 3,
) -> np.ndarray:
    """
    Apply WPE dereverberation to a multi-channel STFT.

    Parameters
    ----------
    stft : np.ndarray
        Complex STFT, shape ``(n_channels, n_freq, n_frames)``.
    taps : int
        Number of filter taps for WPE.
    delay : int
        Prediction delay in frames.

    Returns
    -------
    np.ndarray
        Dereverberated STFT, same shape as input.

    TODO
    ----
    - Implement using ``nara_wpe.wpe`` or ``nara_wpe.wpe_v8``.
    - Handle edge cases (very short signals, single channel).
    """
    # TODO: Implement WPE dereverberation
    logger.warning("apply_wpe() is a placeholder – returning input unchanged.")
    return stft.copy()


def main() -> None:
    """Entry point for step 00 (WPE)."""
    ensure_output_dirs()

    enh_cfg = CFG.get("enhancement", {})
    use_wpe = enh_cfg.get("use_wpe", False)

    stft_path = INTERMEDIATE_DIR / "mixture_stft.npy"

    if stft_path.exists():
        stft = np.load(str(stft_path))
        logger.info("Loaded STFT: %s", stft.shape)
    else:
        logger.warning("STFT not found at %s – creating dummy.", stft_path)
        stft = np.zeros((4, 513, 100), dtype=np.complex64)

    if use_wpe:
        taps = enh_cfg.get("wpe_taps", 10)
        delay = enh_cfg.get("wpe_delay", 3)
        stft = apply_wpe(stft, taps=taps, delay=delay)
        logger.info("WPE applied (taps=%d, delay=%d).", taps, delay)
    else:
        logger.info("WPE disabled in config – skipping.")

    # Save (possibly dereverberated) STFT for next steps
    out_path = INTERMEDIATE_DIR / "mixture_stft_wpe.npy"
    np.save(str(out_path), stft)
    logger.info("Saved → %s", out_path)
    logger.info("Step 00 (WPE) complete.")


if __name__ == "__main__":
    main()
