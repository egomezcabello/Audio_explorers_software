#!/usr/bin/env python3
"""
run_all.py – Execute the complete Member 4 (Fusion) pipeline.

Runs steps 00 → 01 → 02 → 03 sequentially.
(Step 04 is the end-to-end wrapper and is NOT included here to avoid
infinite recursion.)

Usage
-----
    python -m src.member4_fusion.run_all
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from dataclasses import asdict

from src.common.json_schema import SceneSummary, load_json, save_json
from src.common.logging_utils import setup_logging
from src.common.paths import FINAL_DIR, ensure_output_dirs

logger = setup_logging("member4_fusion.run_all")


def main() -> None:
    """Run all Member 4 steps in order (excluding end-to-end)."""
    steps = [
        ("00  Structure check",     "src.member4_fusion.00_project_structure_and_config"),
        ("01  Merge candidates",    "src.member4_fusion.01_merge_candidates"),
        ("02  Score TOI",           "src.member4_fusion.02_score_talker_of_interest"),
        ("03  Make figures",        "src.member4_fusion.03_make_figures"),
    ]

    logger.info("=" * 60)
    logger.info("Member 4 – Fusion pipeline  (4 steps)")
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

    # ── Final scene summary ────────────────────────────────────────────
    ensure_output_dirs()
    merged_path = FINAL_DIR / "merged_scene.json"
    if merged_path.exists():
        scene = load_json(merged_path)
        # Write the definitive final summary
        final_path = FINAL_DIR / "final_scene_summary.json"
        with open(final_path, "w", encoding="utf-8") as fh:
            json.dump(scene, fh, indent=2, ensure_ascii=False)
        logger.info("Final scene summary → %s", final_path)

    logger.info("=" * 60)
    logger.info("Member 4 pipeline complete ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
