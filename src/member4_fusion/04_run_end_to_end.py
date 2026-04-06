#!/usr/bin/env python3
"""
04_run_end_to_end.py – Full end-to-end pipeline (all 4 members).

Sequentially invokes every member's ``run_all.main()`` to produce the
complete scene analysis from raw WAV files to final outputs.

Usage
-----
    python -m src.member4_fusion.04_run_end_to_end
"""

from __future__ import annotations

import importlib
import sys
import traceback

from src.common.logging_utils import setup_logging

logger = setup_logging("pipeline.end_to_end")


PIPELINE_STEPS = [
    ("Member 1 – DoA",         "src.member1_doa.run_all"),
    ("Member 2 – Enhancement", "src.member2_enhance.run_all"),
    ("Member 3 – Analysis",    "src.member3_analysis.run_all"),
    ("Member 4 – Fusion",      "src.member4_fusion.run_all"),
]


def main() -> None:
    """Run the full pipeline end-to-end."""
    logger.info("=" * 70)
    logger.info("  FULL END-TO-END PIPELINE")
    logger.info("=" * 70)

    for label, module_name in PIPELINE_STEPS:
        logger.info("━━━━ %s ━━━━", label)
        try:
            mod = importlib.import_module(module_name)
            mod.main()
        except SystemExit:
            logger.error("Pipeline aborted during '%s'.", label)
            sys.exit(1)
        except Exception as exc:
            logger.error("Pipeline failed at '%s': %s", label, exc)
            traceback.print_exc()
            sys.exit(1)

    logger.info("=" * 70)
    logger.info("  END-TO-END PIPELINE COMPLETE ✓")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
