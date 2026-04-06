"""
src.common.io_utils – Audio I/O helpers.

Thin wrappers around ``soundfile`` that enforce the project's sample-rate
and channel-order conventions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from src.common.constants import CHANNEL_ORDER, N_CHANNELS, SAMPLE_RATE

logger = logging.getLogger(__name__)


def load_multichannel_wav(
    path: Path,
    expected_sr: int = SAMPLE_RATE,
    expected_channels: int = N_CHANNELS,
) -> Tuple[np.ndarray, int]:
    """
    Load a multi-channel WAV file and validate its shape.

    Parameters
    ----------
    path : Path
        Path to the ``.wav`` file.
    expected_sr : int
        Expected sample rate (raises on mismatch).
    expected_channels : int
        Expected number of channels (raises on mismatch).

    Returns
    -------
    audio : np.ndarray
        Shape ``(n_samples, n_channels)`` – float64.
    sr : int
        Sample rate.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If sample rate or channel count doesn't match expectations.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = sf.read(str(path), dtype="float64")

    if sr != expected_sr:
        raise ValueError(
            f"Sample rate mismatch: expected {expected_sr}, got {sr} in {path.name}"
        )

    if audio.ndim == 1:
        raise ValueError(
            f"Expected {expected_channels}-channel audio but got mono in {path.name}"
        )

    if audio.shape[1] != expected_channels:
        raise ValueError(
            f"Channel count mismatch: expected {expected_channels}, "
            f"got {audio.shape[1]} in {path.name}"
        )

    logger.info(
        "Loaded %s – %d samples × %d channels @ %d Hz  (%.2f s)",
        path.name,
        audio.shape[0],
        audio.shape[1],
        sr,
        audio.shape[0] / sr,
    )
    return audio, sr


def save_mono_wav(
    path: Path,
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> None:
    """
    Save a single-channel (mono) WAV file.

    Parameters
    ----------
    path : Path
        Destination file path (parent dirs will be created).
    audio : np.ndarray
        1-D waveform array.
    sr : int
        Sample rate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    logger.info("Saved %s (%.2f s)", path.name, len(audio) / sr)


def channel_label(index: int) -> str:
    """Return the human-readable label for a channel index (0-based)."""
    return CHANNEL_ORDER[index]
