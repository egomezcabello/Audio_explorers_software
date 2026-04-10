#!/usr/bin/env python3
"""
run_all.py – Execute the complete Member 2 (Enhancement) pipeline.

Runs steps 00 → 01 → 02 → 03 sequentially.

Usage
-----
    python -m src.member2_enhance.run_all
"""

from __future__ import annotations

import importlib
import sys
import traceback

from src.common.logging_utils import setup_logging

logger = setup_logging("member2_enhance.run_all")


def main() -> None:
    """Run all Member 2 steps in order."""
    steps = [
        ("00  WPE (optional)", "src.member2_enhance.00_wpe_optional"),
        ("01  Build masks",     "src.member2_enhance.01_build_masks_from_tracks"),
        ("02  MVDR beamform",   "src.member2_enhance.02_mvdr_beamform"),
        ("03  Post-filter & export", "src.member2_enhance.03_postfilter_and_export"),
    ]

    logger.info("=" * 60)
    logger.info("[member2] Enhancement pipeline  (4 steps)")
    logger.info("=" * 60)

    for label, module_name in steps:
        logger.info("──── %s ────", label)
        try:
            mod = importlib.import_module(module_name)
            mod.main()
        except Exception as exc:
            logger.error("[member2] Step '%s' failed: %s", label, exc)
            traceback.print_exc()
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("[member2] pipeline complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
