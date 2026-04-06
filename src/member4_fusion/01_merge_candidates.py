#!/usr/bin/env python3
"""
01_merge_candidates.py – Merge candidate data from all upstream members.

Reads:
  - DoA tracks    (outputs/doa/doa_tracks.json)
  - Analysis JSON (outputs/analysis/spkXX_analysis.json)
  - Enhanced WAVs (outputs/separated/spkXX_enhanced.wav)

Produces a single merged ``SceneSummary`` with fully populated candidate
records.

TODO:
    - Merge DoA track information into each candidate.
    - Attach analysis results (VAD, language, transcript, pitch).
    - Validate all cross-references.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.common.json_schema import (
    Candidate,
    CandidateOutputs,
    SceneSummary,
    load_json,
    save_json,
)
from src.common.logging_utils import setup_logging
from src.common.paths import (
    ANALYSIS_DIR,
    DOA_DIR,
    FINAL_DIR,
    SEPARATED_DIR,
    ensure_output_dirs,
)

logger = setup_logging(__name__)


def merge_candidates() -> SceneSummary:
    """
    Load all upstream outputs and merge into a single SceneSummary.

    Returns
    -------
    SceneSummary
        Fully merged scene summary with candidate details.

    TODO
    ----
    - Implement robust merging with error handling.
    - Cross-validate DoA track IDs with analysis file IDs.
    """
    summary = SceneSummary()

    # Load DoA tracks
    tracks_path = DOA_DIR / "doa_tracks.json"
    if tracks_path.exists():
        raw = load_json(tracks_path)
        raw_candidates = raw.get("candidates", [])
    else:
        logger.warning("No DoA tracks found – starting with empty candidate list.")
        raw_candidates = []

    for rc in raw_candidates:
        cid = rc.get("id", "spk00")

        # Build output paths
        outputs = CandidateOutputs(
            enhanced_wav=str(SEPARATED_DIR / f"{cid}_enhanced.wav"),
            analysis_json=str(ANALYSIS_DIR / f"{cid}_analysis.json"),
            transcript_txt=str(ANALYSIS_DIR / f"{cid}_transcript.txt"),
        )

        cand = Candidate(
            id=cid,
            doa_track=rc.get("doa_track", []),
            active_segments=rc.get("active_segments", []),
            outputs=outputs,
        )
        summary.candidates.append(cand)

    logger.info("Merged %d candidate(s).", len(summary.candidates))
    return summary


def main() -> None:
    """Entry point for step 01 (merge)."""
    ensure_output_dirs()

    summary = merge_candidates()
    out_path = FINAL_DIR / "merged_scene.json"
    save_json(summary, out_path)
    logger.info("Merged scene saved → %s", out_path)
    logger.info("Step 01 (merge) complete.")


if __name__ == "__main__":
    main()
