"""
src.common.plotting – Shared Matplotlib plotting helpers.

Provides reusable figure-creation functions so that report figures have a
consistent style across all members.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI
import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a Matplotlib figure to *path*, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
