#!/usr/bin/env python3
"""
run_all.py – Execute the complete Member 1 (DoA) pipeline.
===========================================================

Runs steps 00 → 01 → 02 → 03 → 06 sequentially:

  00  Load & STFT       — for **all** WAVs
  01  Calibrate         — on ``example_mixture.wav`` **only**
  02  DoA estimate      — per tag (uses calibration from step 01)
  03  Track & cluster   — per tag
  06  Visualize         — per tag

Sweep / tuning scripts (04, 05) are run separately.

If any step fails, the pipeline stops and reports the error.

Usage
-----
    python -m src.member1_doa.run_all          # from project root
    python src/member1_doa/run_all.py          # direct invocation
    python -m src.member1_doa.run_all --tags mixture   # one tag
"""

from __future__ import annotations

import importlib
import sys
import traceback

from src.common.config import CFG
from src.common.logging_utils import setup_logging

logger = setup_logging("member1_doa.run_all")


def _run_step(label: str, module_name: str, **kwargs) -> None:
    """Import and run a single pipeline step, aborting on failure."""
    extra = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    logger.info("──── %s (%s) ────", label, extra or "defaults")
    try:
        mod = importlib.import_module(module_name)
        mod.main(**kwargs)
    except Exception as exc:
        logger.error("Step '%s' failed: %s", label, exc)
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    """Run all Member 1 steps in order for every available input."""
    import argparse
    parser = argparse.ArgumentParser(description="Run Member 1 DoA pipeline")
    parser.add_argument(
        "--tags", nargs="+", default=None,
        help="Input tags to process (default: auto-detect from intermediate STFTs)",
    )
    parser.add_argument(
        "--sweep", action="store_true", default=False,
        help="Run parameter sweep / tuning instead of the normal pipeline",
    )
    args = parser.parse_args()

    if args.sweep:
        import importlib, sys
        mod = importlib.import_module("src.member1_doa.04_sweep_tuning")
        logger.info("Delegating to sweep / tuning script ...")
        sys.argv = [sys.argv[0]]  # reset argv so sweep's argparse works
        mod.main()
        return

    logger.info("=" * 60)
    logger.info("Member 1 – DoA pipeline")
    logger.info("=" * 60)

    # ── Step 00: Load & STFT (all WAVs) ────────────────────────────────
    _run_step("00  Load & STFT", "src.member1_doa.00_load_and_stft")

    # ── Step 01: Calibrate (example_mixture only) ──────────────────────
    calib_tag = CFG.get("doa", {}).get("calibration_source", "example")
    _run_step("01  Calibrate", "src.member1_doa.01_calibrate_templates",
              tag=calib_tag)

    # ── Determine tags to process for DoA / tracking ───────────────────
    if args.tags:
        tags = list(args.tags)
    else:
        from src.common.paths import INTERMEDIATE_DIR
        stft_files = sorted(INTERMEDIATE_DIR.glob("*_stft.npy"))
        tags = [f.stem.replace("_stft", "") for f in stft_files]
        logger.info("Auto-detected input tags: %s", tags)

    # Always include the calibration tag (example) so that example
    # validation results are printed even when running only --tags mixture.
    if calib_tag not in tags:
        tags = [calib_tag] + tags

    # ── Steps 02–03, 06 for each tag ─────────────────────────────────────
    for tag in tags:
        logger.info("═" * 60)
        logger.info("Processing tag: %s", tag)
        logger.info("═" * 60)
        _run_step("02  DoA estimate",    "src.member1_doa.02_doa_estimate",
                  tag=tag)
        _run_step("03  Track & cluster", "src.member1_doa.03_track_and_cluster",
                  tag=tag)
        _run_step("06  Visualize results",
                  "src.member1_doa.06_visualize_results", tag=tag)

    logger.info("=" * 60)
    logger.info("Member 1 pipeline complete  (%d input(s))", len(tags))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
