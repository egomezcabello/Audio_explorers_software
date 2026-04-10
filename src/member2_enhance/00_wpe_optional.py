#!/usr/bin/env python3
"""
00_wpe_optional.py – Optional WPE dereverberation.
====================================================
First step of the Member 2 (Enhancement) pipeline.

Applies Weighted Prediction Error (WPE) dereverberation to the
multi-channel STFT if enabled in ``config.yaml``.  If ``nara_wpe`` is
not installed or fails at runtime, falls back to an unmodified copy.

STFT convention: ``(n_channels, n_freq, n_frames)`` — same as Member 1.

Outputs
-------
- ``outputs/intermediate/mixture_stft_wpe.npy``
"""

from __future__ import annotations

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
    Apply offline WPE dereverberation using ``nara_wpe``.

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
    """
    try:
        from nara_wpe.wpe import wpe as nara_wpe_func
    except ImportError:
        logger.warning("[member2][wpe] nara_wpe not installed — skipping.")
        return stft.copy()

    try:
        # nara_wpe expects shape (n_channels, n_freq, n_frames)
        dereverberated = nara_wpe_func(
            stft,
            taps=taps,
            delay=delay,
            iterations=3,
        )
        return dereverberated
    except Exception as exc:
        logger.warning("[member2][wpe] nara_wpe failed (%s) — skipping.", exc)
        return stft.copy()


def main() -> None:
    """Entry point for step 00 (WPE)."""
    ensure_output_dirs()

    enh_cfg = CFG.get("enhancement", {})
    use_wpe = enh_cfg.get("use_wpe", False)

    stft_path = INTERMEDIATE_DIR / "mixture_stft.npy"
    if not stft_path.exists():
        raise FileNotFoundError(
            f"[member2][wpe] STFT not found: {stft_path} — run Member 1 first."
        )

    stft = np.load(str(stft_path))
    logger.info("[member2][wpe] loaded STFT %s from %s", stft.shape, stft_path.name)

    if use_wpe:
        taps = enh_cfg.get("wpe_taps", 10)
        delay = enh_cfg.get("wpe_delay", 3)
        stft_out = apply_wpe(stft, taps=taps, delay=delay)
        changed = not np.array_equal(stft, stft_out)
        logger.info("[member2][wpe] WPE %s (taps=%d, delay=%d)",
                    "applied" if changed else "no-op (fallback)", taps, delay)
    else:
        stft_out = stft.copy()
        logger.info("[member2][wpe] disabled in config — pass-through copy")

    out_path = INTERMEDIATE_DIR / "mixture_stft_wpe.npy"
    np.save(str(out_path), stft_out)
    logger.info("[member2][wpe] saved → %s  shape=%s", out_path.name, stft_out.shape)


if __name__ == "__main__":
    main()
