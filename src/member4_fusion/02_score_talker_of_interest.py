#!/usr/bin/env python3
"""
02_score_talker_of_interest.py – Score candidates and select TOI.

Reads the merged scene summary and scores each candidate based on
configurable weights (DoA stability, speech duration, language match,
SNR estimate).  The highest-scoring candidate is designated as the
"talker of interest" (TOI).

TODO:
    - Implement scoring functions for each criterion.
    - Combine scores with configurable weights.
    - Copy / symlink the TOI enhanced WAV to outputs/final/.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.common.config import CFG
from src.common.constants import SAMPLE_RATE
from src.common.json_schema import load_json, save_json
from src.common.logging_utils import setup_logging
from src.common.paths import ANALYSIS_DIR, FINAL_DIR, SEPARATED_DIR, ensure_output_dirs

logger = setup_logging(__name__)


@dataclass
class CandidateScore:
    """Score breakdown for one candidate."""
    candidate_id: str = ""
    doa_stability: float = 0.0
    speech_duration: float = 0.0
    language_match: float = 0.0
    snr_estimate: float = 0.0
    total_score: float = 0.0


def score_doa_stability(doa_track: list) -> float:
    """
    Score DoA stability (low variance → high score).

    TODO: Implement actual DoA variance computation.
    """
    # TODO: Implement
    return 0.5  # placeholder neutral score


def score_speech_duration(analysis: Dict[str, Any]) -> float:
    """
    Score based on total speech duration from VAD segments.

    TODO: Implement – longer speech → higher score (normalised).
    """
    # TODO: Implement
    return 0.5


def score_language_match(analysis: Dict[str, Any], target_lang: str = "en") -> float:
    """
    Score based on language match to expected target.

    TODO: Implement – exact match → 1.0, else lower.
    """
    # TODO: Implement
    return 0.5


def score_snr(candidate_id: str) -> float:
    """
    Estimate SNR of the enhanced signal.

    TODO: Implement SNR estimation.
    """
    # TODO: Implement
    return 0.5


def compute_total_score(scores: CandidateScore, weights: Dict[str, float]) -> float:
    """Weighted sum of individual scores."""
    total = (
        weights.get("doa_stability", 0.25) * scores.doa_stability
        + weights.get("speech_duration", 0.25) * scores.speech_duration
        + weights.get("language_match", 0.25) * scores.language_match
        + weights.get("snr_estimate", 0.25) * scores.snr_estimate
    )
    return total


def main() -> None:
    """Entry point for step 02 (scoring)."""
    ensure_output_dirs()

    fusion_cfg = CFG.get("fusion", {})
    weights = fusion_cfg.get("score_weights", {})

    # Load merged scene
    merged_path = FINAL_DIR / "merged_scene.json"
    if merged_path.exists():
        scene = load_json(merged_path)
        candidates = scene.get("candidates", [])
    else:
        logger.warning("Merged scene not found – nothing to score.")
        candidates = []

    scores: List[CandidateScore] = []

    for cand in candidates:
        cid = cand.get("id", "spk00")
        doa_track = cand.get("doa_track", [])

        # Load analysis if available
        analysis_path = ANALYSIS_DIR / f"{cid}_analysis.json"
        analysis = {}
        if analysis_path.exists():
            with open(analysis_path, "r", encoding="utf-8") as fh:
                analysis = json.load(fh)

        cs = CandidateScore(candidate_id=cid)
        cs.doa_stability = score_doa_stability(doa_track)
        cs.speech_duration = score_speech_duration(analysis)
        cs.language_match = score_language_match(analysis)
        cs.snr_estimate = score_snr(cid)
        cs.total_score = compute_total_score(cs, weights)
        scores.append(cs)
        logger.info("  %s → total_score=%.3f", cid, cs.total_score)

    # Select talker of interest
    if scores:
        best = max(scores, key=lambda s: s.total_score)
        logger.info("Talker of interest: %s (score=%.3f)", best.candidate_id, best.total_score)

        # Copy enhanced WAV to final
        src_wav = SEPARATED_DIR / f"{best.candidate_id}_enhanced.wav"
        dst_wav = FINAL_DIR / "talker_of_interest.wav"
        if src_wav.exists():
            shutil.copy2(str(src_wav), str(dst_wav))
            logger.info("Copied TOI WAV → %s", dst_wav)
        else:
            logger.warning("Enhanced WAV for TOI not found at %s", src_wav)

    # Save scores
    from dataclasses import asdict
    scores_path = FINAL_DIR / "candidate_scores.json"
    with open(scores_path, "w", encoding="utf-8") as fh:
        json.dump([asdict(s) for s in scores], fh, indent=2)
    logger.info("Scores saved → %s", scores_path)
    logger.info("Step 02 (scoring) complete.")


if __name__ == "__main__":
    main()
