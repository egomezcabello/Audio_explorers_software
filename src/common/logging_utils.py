"""
src.common.logging_utils – Shared logging configuration.

Call ``setup_logging()`` once at the start of each member script to get
consistent, timestamped log output across the whole pipeline.
"""

from __future__ import annotations

import logging
import sys

from src.common.config import CFG


def setup_logging(name: str = "audio_explorers") -> logging.Logger:
    """
    Configure and return a logger with the project-wide format and level.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_cfg = CFG.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt = log_cfg.get(
        "format",
        "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    )

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
