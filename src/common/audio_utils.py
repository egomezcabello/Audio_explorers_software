"""
src.common.audio_utils – General-purpose audio helpers.

Provides lightweight wrappers for common operations such as normalisation,
RMS computation, and segmenting.  Heavier DSP lives in dedicated member
modules.
"""

from __future__ import annotations

import numpy as np


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Peak-normalize an audio array so the maximum absolute value equals
    *target_peak*.

    Parameters
    ----------
    audio : np.ndarray
        Input waveform (any shape).
    target_peak : float
        Desired peak amplitude (0 < target_peak ≤ 1.0).

    Returns
    -------
    np.ndarray
        Normalized waveform, same shape as input.
    """
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio * (target_peak / peak)


def rms(audio: np.ndarray) -> float:
    """Return the root-mean-square energy of *audio*."""
    return float(np.sqrt(np.mean(audio ** 2)))


def rms_db(audio: np.ndarray, ref: float = 1.0) -> float:
    """Return RMS in dB relative to *ref*."""
    r = rms(audio)
    if r == 0:
        return -np.inf
    return float(20.0 * np.log10(r / ref))


def segment_audio(
    audio: np.ndarray,
    sr: int,
    seg_length_s: float = 1.0,
) -> list[np.ndarray]:
    """
    Split *audio* into fixed-length segments (last segment may be shorter).

    Parameters
    ----------
    audio : np.ndarray
        1-D waveform.
    sr : int
        Sample rate.
    seg_length_s : float
        Segment length in seconds.

    Returns
    -------
    list[np.ndarray]
        List of 1-D arrays.
    """
    seg_len = int(sr * seg_length_s)
    return [audio[i : i + seg_len] for i in range(0, len(audio), seg_len)]
