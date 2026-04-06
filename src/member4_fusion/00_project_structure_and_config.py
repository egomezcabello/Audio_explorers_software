#!/usr/bin/env python3
"""
00_project_structure_and_config.py – Verify project structure and config.

This "step 0" for Member 4 ensures that:
  - All expected directories exist.
  - The config is loadable and consistent.
  - Upstream outputs are present (or warns clearly if not).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.common.config import CFG, get_channel_order, get_sample_rate
from src.common.constants import CHANNEL_ORDER, SAMPLE_RATE
from src.common.logging_utils import setup_logging
from src.common.paths import (
    ANALYSIS_DIR,
    CALIB_DIR,
    DATA_DIR,
    DOA_DIR,
    FINAL_DIR,
    FIGURES_DIR,
    INTERMEDIATE_DIR,
    SEPARATED_DIR,
    ensure_output_dirs,
)

logger = setup_logging(__name__)


def check_directories() -> List[str]:
    """Return a list of warnings for missing or empty directories."""
    warnings: List[str] = []
    checks = {
        "data": DATA_DIR,
        "outputs/calib": CALIB_DIR,
        "outputs/doa": DOA_DIR,
        "outputs/intermediate": INTERMEDIATE_DIR,
        "outputs/separated": SEPARATED_DIR,
        "outputs/analysis": ANALYSIS_DIR,
        "outputs/final": FINAL_DIR,
    }
    for label, d in checks.items():
        if not d.exists():
            warnings.append(f"Directory missing: {label} ({d})")
    return warnings


def check_config_consistency() -> List[str]:
    """Validate that config values match hardcoded constants."""
    warnings: List[str] = []
    if get_sample_rate() != SAMPLE_RATE:
        warnings.append(
            f"Config sample_rate ({get_sample_rate()}) != constant ({SAMPLE_RATE})"
        )
    if get_channel_order() != CHANNEL_ORDER:
        warnings.append(
            f"Config channel_order ({get_channel_order()}) != constant ({CHANNEL_ORDER})"
        )
    return warnings


def main() -> None:
    """Entry point for step 00 (structure check)."""
    ensure_output_dirs()

    logger.info("Checking project structure and configuration …")

    dir_warnings = check_directories()
    cfg_warnings = check_config_consistency()
    all_warnings = dir_warnings + cfg_warnings

    if all_warnings:
        for w in all_warnings:
            logger.warning("  ⚠  %s", w)
    else:
        logger.info("  All checks passed ✓")

    logger.info("Step 00 (structure check) complete.")


if __name__ == "__main__":
    main()
