#!/usr/bin/env python3
"""
00_load_and_stft.py – Load 4-channel WAV files and compute STFT.
=================================================================
First step of the Member 1 (DoA) pipeline.

What it does
------------
1. Load ``data/example_mixture.wav`` and ``data/mixture.wav``.
2. **Verify** each file: exists, sample rate == 44 100 Hz, exactly 4 channels.
3. Compute the multi-channel STFT for each file.
4. Save the complex STFT tensors to ``outputs/intermediate/``.

STFT tensor convention  (used everywhere in Member 1)
-----------------------------------------------------
    X.shape == (n_channels, n_freq, n_frames)

    axis 0  →  channel index   [LF=0, LR=1, RF=2, RR=3]
    axis 1  →  frequency bin   [0 … n_fft/2]
    axis 2  →  time frame      [0 … T-1]

This convention is consistent across steps 00 → 01 → 02 → 03.

Outputs
-------
- ``outputs/intermediate/example_stft.npy``   (from example_mixture.wav)
- ``outputs/intermediate/mixture_stft.npy``    (from mixture.wav)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.common.config import get_stft_params
from src.common.constants import CHANNEL_ORDER, N_CHANNELS, SAMPLE_RATE
from src.common.io_utils import load_multichannel_wav
from src.common.logging_utils import setup_logging
from src.common.paths import (
    EXAMPLE_MIXTURE_WAV,
    INTERMEDIATE_DIR,
    MIXTURE_WAV,
    ensure_output_dirs,
)
from src.common.stft_utils import compute_stft

logger = setup_logging(__name__)


# ── Helpers ────────────────────────────────────────────────────────────

def load_and_verify(wav_path: Path) -> np.ndarray:
    """
    Load a WAV file and verify it matches project conventions.

    Raises
    ------
    FileNotFoundError
        If the WAV file does not exist.
    ValueError
        If the sample rate is not 44 100 Hz or the file does not have
        exactly 4 channels.

    Returns
    -------
    audio : np.ndarray
        Shape ``(n_samples, 4)``, float64.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    audio, sr = load_multichannel_wav(wav_path)

    # load_multichannel_wav already validates sr and n_channels, but we
    # log explicitly so the user always sees what was loaded:
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected sample rate {SAMPLE_RATE}, got {sr} in {wav_path.name}"
        )
    if audio.ndim != 2 or audio.shape[1] != N_CHANNELS:
        raise ValueError(
            f"Expected {N_CHANNELS}-channel audio, got shape {audio.shape} "
            f"in {wav_path.name}"
        )

    logger.info(
        "  Loaded %s: shape=%s, sr=%d, channels=%s",
        wav_path.name, audio.shape, sr, CHANNEL_ORDER,
    )
    return audio


def compute_and_save_stft(audio: np.ndarray, out_path: Path) -> None:
    """
    Compute multi-channel STFT and save the complex tensor.

    The saved array has shape ``(n_channels, n_freq, n_frames)`` — this
    is the convention used across the whole Member 1 pipeline.
    """
    _f, _t, Zxx = compute_stft(audio)
    # Zxx shape from stft_utils: (n_channels, n_freq, n_frames)
    logger.info(
        "  STFT shape: %s  [channels × freq × frames]  —  convention: X[ch, f, t]",
        Zxx.shape,
    )
    np.save(str(out_path), Zxx)
    logger.info("  Saved → %s", out_path)


# ── Entry point ────────────────────────────────────────────────────────

def main(skip_example: bool = False) -> None:
    """
    Load both WAV files, compute STFTs, and save them.

    Parameters
    ----------
    skip_example : bool
        If True, skip processing of example_mixture.wav.
    """
    ensure_output_dirs()

    stft_params = get_stft_params()
    logger.info(
        "STFT params: n_fft=%d, hop=%d, win=%d, window=%s",
        stft_params["n_fft"], stft_params["hop_length"],
        stft_params["win_length"], stft_params["window"],
    )

    # ── mixture.wav ────────────────────────────────────────────────────
    if not MIXTURE_WAV.exists():
        raise FileNotFoundError(
            f"Required file not found: {MIXTURE_WAV}  "
            "— place the challenge WAV in data/"
        )
    logger.info("Loading %s", MIXTURE_WAV)
    audio_mix = load_and_verify(MIXTURE_WAV)
    compute_and_save_stft(audio_mix, INTERMEDIATE_DIR / "mixture_stft.npy")

    # ── example_mixture.wav ────────────────────────────────────────────
    if not skip_example:
        if not EXAMPLE_MIXTURE_WAV.exists():
            raise FileNotFoundError(
                f"Required file not found: {EXAMPLE_MIXTURE_WAV}  "
                "— place the example WAV in data/"
            )
        logger.info("Loading %s", EXAMPLE_MIXTURE_WAV)
        audio_ex = load_and_verify(EXAMPLE_MIXTURE_WAV)
        compute_and_save_stft(audio_ex, INTERMEDIATE_DIR / "example_stft.npy")

    logger.info("Step 00 complete.")


if __name__ == "__main__":
    _p = argparse.ArgumentParser(description="Step 00 – Load WAVs & compute STFTs")
    _p.add_argument("--skip-example", action="store_true",
                    help="Skip processing of example_mixture.wav")
    _args = _p.parse_args()
    main(skip_example=_args.skip_example)
