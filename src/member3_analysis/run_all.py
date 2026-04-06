#!/usr/bin/env python3
"""
run_all.py – Execute the complete Member 3 (Analysis) pipeline.

Runs steps 00 → 01 → 02 → 03 → 04 sequentially.

Usage
-----
    python -m src.member3_analysis.run_all
"""

from __future__ import annotations

import importlib
import sys
import traceback

from src.common.logging_utils import setup_logging

logger = setup_logging("member3_analysis.run_all")


def main() -> None:
    """Run all Member 3 steps in order."""
    steps = [
        ("00  VAD & turns",      "src.member3_analysis.00_vad_and_turns"),
        ("01  Language ID",      "src.member3_analysis.01_language_id"),
        ("02  ASR (Whisper)",    "src.member3_analysis.02_asr_whisper"),
        ("03  Pitch / voice",    "src.member3_analysis.03_pitch_voice_type"),
        ("04  Pack results",     "src.member3_analysis.04_pack_results"),
    ]

    logger.info("=" * 60)
    logger.info("Member 3 – Analysis pipeline  (5 steps)")
    logger.info("=" * 60)

    for label, module_name in steps:
        logger.info("──── %s ────", label)
        try:
            mod = importlib.import_module(module_name)
            mod.main()
        except Exception as exc:
            logger.error("Step '%s' failed: %s", label, exc)
            traceback.print_exc()
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Member 3 pipeline complete ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
