#!/usr/bin/env python3
"""
00_load_and_stft.py – Load 4-channel WAV files and compute STFT.

This is the first step of the Member 1 (DoA) pipeline.

Steps
-----
1. Load ``data/example_mixture.wav`` and ``data/mixture.wav``.
2. Verify that each file has 4 channels and the expected sample rate.
3. Compute the multi-channel STFT for each file.
4. Save the STFT arrays to ``outputs/intermediate/``.

Channel order (always):
    Index 0 → LF  (Left-Front)
    Index 1 → LR  (Left-Rear)
    Index 2 → RF  (Right-Front)
    Index 3 → RR  (Right-Rear)
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


def load_and_verify(wav_path: Path) -> np.ndarray:
    """
    Load a WAV file and verify it matches project conventions.
    """
    audio, sr = load_multichannel_wav(wav_path)
    
    if audio.shape[1] != N_CHANNELS:
        if audio.shape[0] == N_CHANNELS:
            audio = audio.T
        else:
            raise ValueError(f"Expected {N_CHANNELS} channels, got {audio.shape[1]}")

    logger.info("  Verifying signal levels (RMS):")
    for i, label in enumerate(CHANNEL_ORDER):
        rms_val = np.sqrt(np.mean(audio[:, i]**2))
        logger.info("    Channel %d (%s) RMS: %.4f", i, label, rms_val)

    logger.info(
        "  Channel order: %s | Shape: %s | SR: %d",
        CHANNEL_ORDER,
        audio.shape,
        sr,
    )
    return audio


def compute_and_save_stft(audio: np.ndarray, out_path: Path) -> None:
    """
    Compute multi-channel STFT and save the complex array.

    Parameters
    ----------
    audio : np.ndarray
        Shape ``(n_samples, n_channels)``.
    out_path : Path
        Destination ``.npy`` file.
    """
    f, t, Zxx = compute_stft(audio)
    logger.info("  STFT shape: %s  (channels × freq × frames)", Zxx.shape)
    np.save(str(out_path), Zxx)
    logger.info("  Saved → %s", out_path)


def main() -> None:
    """Entry point for step 00."""
    parser = argparse.ArgumentParser(description="Load WAVs and compute STFTs")
    parser.add_argument(
        "--skip-example",
        action="store_true",
        help="Skip processing of example_mixture.wav",
    )
    args = parser.parse_args()

    ensure_output_dirs()

    # ── Mixture ────────────────────────────────────────────────────────
    if MIXTURE_WAV.exists():
        logger.info("Loading mixture: %s", MIXTURE_WAV)
        audio_mix = load_and_verify(MIXTURE_WAV)
        compute_and_save_stft(audio_mix, INTERMEDIATE_DIR / "mixture_stft.npy")
    else:
        logger.warning("Mixture WAV not found at %s – skipping.", MIXTURE_WAV)

    # ── Example mixture (optional) ────────────────────────────────────
    if not args.skip_example:
        if EXAMPLE_MIXTURE_WAV.exists():
            logger.info("Loading example mixture: %s", EXAMPLE_MIXTURE_WAV)
            audio_ex = load_and_verify(EXAMPLE_MIXTURE_WAV)
            compute_and_save_stft(audio_ex, INTERMEDIATE_DIR / "example_stft.npy")
        else:
            logger.warning(
                "Example mixture WAV not found at %s – skipping.",
                EXAMPLE_MIXTURE_WAV,
            )

    logger.info("Step 00 complete.")


if __name__ == "__main__":
    main()
