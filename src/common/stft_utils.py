"""
src.common.stft_utils – STFT / iSTFT convenience wrappers.

Wraps ``scipy.signal.stft`` and ``istft`` using the project's default
parameters from ``config.yaml``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import istft as _istft
from scipy.signal import stft as _stft

from src.common.config import get_stft_params
from src.common.constants import SAMPLE_RATE


def compute_stft(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    **overrides,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform of a multi- or single-channel
    signal.

    Parameters
    ----------
    audio : np.ndarray
        Shape ``(n_samples,)`` for mono or ``(n_samples, n_channels)`` for
        multi-channel.
    sr : int
        Sample rate.
    **overrides
        Override any key from the STFT config (n_fft, hop_length, …).

    Returns
    -------
    f : np.ndarray
        Frequency bin centres.
    t : np.ndarray
        Time-frame centres.
    Zxx : np.ndarray
        Complex STFT matrix.  Shape ``(n_freq, n_frames)`` for mono or
        ``(n_channels, n_freq, n_frames)`` for multi-channel.
    """
    params = get_stft_params()
    params.update(overrides)

    n_fft = params["n_fft"]
    hop = params["hop_length"]
    win_len = params["win_length"]
    window = params["window"]

    if audio.ndim == 1:
        f, t, Zxx = _stft(audio, fs=sr, window=window, nperseg=win_len,
                           noverlap=win_len - hop, nfft=n_fft)
        return f, t, Zxx

    # Multi-channel: stack along a new leading axis
    stfts = []
    for ch in range(audio.shape[1]):
        f, t, Zxx_ch = _stft(audio[:, ch], fs=sr, window=window,
                              nperseg=win_len, noverlap=win_len - hop,
                              nfft=n_fft)
        stfts.append(Zxx_ch)
    return f, t, np.stack(stfts, axis=0)


def compute_istft(
    Zxx: np.ndarray,
    sr: int = SAMPLE_RATE,
    **overrides,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the inverse STFT.

    Parameters
    ----------
    Zxx : np.ndarray
        Complex STFT matrix – ``(n_freq, n_frames)`` for mono.
    sr : int
        Sample rate.
    **overrides
        Override any key from the STFT config.

    Returns
    -------
    t : np.ndarray
        Time vector.
    audio : np.ndarray
        Reconstructed waveform.
    """
    params = get_stft_params()
    params.update(overrides)

    win_len = params["win_length"]
    hop = params["hop_length"]
    n_fft = params["n_fft"]
    window = params["window"]

    t, x = _istft(Zxx, fs=sr, window=window, nperseg=win_len,
                   noverlap=win_len - hop, nfft=n_fft)
    return t, x
