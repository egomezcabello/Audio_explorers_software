"""
src.common.plotting – Shared Matplotlib plotting helpers.

Provides reusable figure-creation functions so that report figures have a
consistent style across all members.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI
import matplotlib.pyplot as plt
import numpy as np

# ── Project-wide Matplotlib style ──────────────────────────────────────
STYLE: dict = {
    "figure.figsize": (10, 4),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
}
plt.rcParams.update(STYLE)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a Matplotlib figure to *path*, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_waveform(
    audio: np.ndarray,
    sr: int,
    title: str = "Waveform",
    channel_labels: Optional[list[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot a multi-channel waveform.

    Parameters
    ----------
    audio : np.ndarray
        Shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
    sr : int
        Sample rate.
    title : str
        Figure title.
    channel_labels : list[str] or None
        Label for each channel subplot.
    save_path : Path or None
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    n_ch = audio.shape[1]
    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 2 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    t = np.arange(audio.shape[0]) / sr
    for i, ax in enumerate(axes):
        ax.plot(t, audio[:, i], linewidth=0.4)
        label = channel_labels[i] if channel_labels else f"Ch {i}"
        ax.set_ylabel(label)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(title)
    fig.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)

    return fig


def plot_spectrogram(
    Zxx: np.ndarray,
    sr: int,
    hop_length: int,
    title: str = "Spectrogram",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot a magnitude spectrogram (dB scale).

    Parameters
    ----------
    Zxx : np.ndarray
        Complex STFT matrix, shape ``(n_freq, n_frames)``.
    sr : int
        Sample rate.
    hop_length : int
        STFT hop length.
    title : str
        Figure title.
    save_path : Path or None
        If given, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mag_db = 20.0 * np.log10(np.abs(Zxx) + 1e-10)
    n_frames = mag_db.shape[1]

    fig, ax = plt.subplots()
    im = ax.imshow(
        mag_db,
        aspect="auto",
        origin="lower",
        extent=[0, n_frames * hop_length / sr, 0, sr / 2],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()

    if save_path is not None:
        save_figure(fig, save_path)

    return fig
