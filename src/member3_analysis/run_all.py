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
from src.common.paths import ANALYSIS_DIR

logger = setup_logging("member3_analysis.run_all")


def _clean_stale_analysis() -> None:
    """Remove per-candidate analysis artefacts from previous runs.

    This prevents stale results (from candidates that no longer exist)
    from leaking into downstream steps when M3 is re-run after M2
    produced a different candidate set.
    """
    patterns = ["*_vad.npy", "*_transcript.txt", "*_analysis.json"]
    removed = 0
    for pat in patterns:
        for f in ANALYSIS_DIR.glob(pat):
            f.unlink()
            removed += 1
    # Also clean aggregate files that accumulate entries
    for agg in ["language_id_results.json", "pitch_results.json"]:
        p = ANALYSIS_DIR / agg
        if p.exists():
            p.unlink()
            removed += 1
    if removed:
        logger.info("Cleaned %d stale analysis file(s)", removed)


def _print_transcripts() -> None:
    """Print all transcripts and pitch results after analysis."""
    tx_files = sorted(ANALYSIS_DIR.glob("*_transcript.txt"))
    if not tx_files:
        return

    logger.info("─" * 60)
    logger.info("Transcripts")
    logger.info("─" * 60)
    for tx in tx_files:
        cid = tx.stem.replace("_transcript", "")
        text = tx.read_text(encoding="utf-8").strip()
        logger.info("  [%s] %s", cid, text)

    # Also print pitch summary if available
    import json
    pitch_path = ANALYSIS_DIR / "pitch_results.json"
    if pitch_path.exists():
        logger.info("─" * 60)
        logger.info("Pitch summary")
        logger.info("─" * 60)
        with open(pitch_path, "r", encoding="utf-8") as fh:
            for item in json.load(fh):
                logger.info("  [%s] F0=%.1f Hz  %s",
                            item.get("candidate_id", "?"),
                            item.get("median_f0_hz", 0),
                            item.get("voice_type", "?"))


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

    _clean_stale_analysis()

    for label, module_name in steps:
        logger.info("──── %s ────", label)
        try:
            mod = importlib.import_module(module_name)
            mod.main()
        except Exception as exc:
            logger.error("Step '%s' failed: %s", label, exc)
            traceback.print_exc()
            sys.exit(1)

    _print_transcripts()

    logger.info("=" * 60)
    logger.info("Member 3 pipeline complete ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
