#!/usr/bin/env python3
"""
run_all.py – Execute the complete Member 1 (DoA) pipeline.

Runs steps 00 → 01 → 02 → 03 sequentially.  If any step fails, the
pipeline stops and reports the error.

Usage
-----
    python -m src.member1_doa.run_all          # from project root
    python src/member1_doa/run_all.py          # direct invocation

If the input WAV files are not present, the pipeline will still run and
produce placeholder outputs so downstream members can test their code.
"""

from __future__ import annotations

import sys
import traceback

from src.common.logging_utils import setup_logging

logger = setup_logging("member1_doa.run_all")


def main() -> None:
    """Run all Member 1 steps in order."""
    steps = [
        ("00  Load & STFT", "src.member1_doa.00_load_and_stft"),
        ("01  Calibrate",   "src.member1_doa.01_calibrate_templates"),
        ("02  DoA estimate", "src.member1_doa.02_doa_estimate"),
        ("03  Track & cluster", "src.member1_doa.03_track_and_cluster"),
    ]

    logger.info("=" * 60)
    logger.info("Member 1 – DoA pipeline  (4 steps)")
    logger.info("=" * 60)

    for label, module_name in steps:
        logger.info("──── %s ────", label)
        try:
            # Import and call the step's main()
            import importlib
            mod = importlib.import_module(module_name)
            mod.main()
        except Exception as exc:
            logger.error("Step '%s' failed: %s", label, exc)
            traceback.print_exc()
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Member 1 pipeline complete ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
