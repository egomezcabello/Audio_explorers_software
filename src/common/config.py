"""
src.common.config – Load and expose the project configuration.

Reads ``config.yaml`` from the project root and makes it available as a
plain dict (``CFG``) or via typed helper accessors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.common.paths import PROJECT_ROOT

_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> Dict[str, Any]:
    """
    Load and return the YAML configuration as a nested dict.

    Parameters
    ----------
    path : Path
        Path to the YAML file.  Defaults to ``<project_root>/config.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# Module-level singleton – import as ``from src.common.config import CFG``
CFG: Dict[str, Any] = load_config()


# ── Typed convenience accessors ────────────────────────────────────────
def get_sample_rate() -> int:
    """Return the configured sample rate (Hz)."""
    return int(CFG["audio"]["sample_rate"])


def get_channel_order() -> List[str]:
    """Return the channel label list, e.g. ['LF', 'LR', 'RF', 'RR']."""
    return list(CFG["audio"]["channel_order"])


def get_stft_params() -> Dict[str, Any]:
    """Return STFT parameters as a dict suitable for ``scipy.signal.stft``."""
    return dict(CFG["stft"])
