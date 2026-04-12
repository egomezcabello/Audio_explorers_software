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

from src.common.config import CFG
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


# ── Deduplication helpers ──────────────────────────────────────────────

def _transcript_similarity(t1: str, t2: str) -> float:
    """Word-level Jaccard similarity between two transcripts."""
    words1 = set(t1.lower().split())
    words2 = set(t2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


def _deduplicate(
    candidates: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Group candidates with similar transcripts; keep best per group."""
    n = len(candidates)
    if n <= 1:
        return candidates

    # Load transcripts
    transcripts: List[str] = []
    for cand in candidates:
        cid = cand.get("id", "")
        tx_path = ANALYSIS_DIR / f"{cid}_transcript.txt"
        tx = tx_path.read_text(encoding="utf-8").strip() if tx_path.exists() else ""
        transcripts.append(tx)

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            sim = _transcript_similarity(transcripts[i], transcripts[j])
            if sim >= threshold:
                union(i, j)
                logger.info("  duplicate: %s ↔ %s (Jaccard=%.2f)",
                            candidates[i].get("id"), candidates[j].get("id"), sim)

    # Group by root
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    deduped: List[Dict[str, Any]] = []
    for members in groups.values():
        if len(members) == 1:
            deduped.append(candidates[members[0]])
        else:
            # Prefer confirmed (spk*) over provisional; then highest score
            best_idx = max(members, key=lambda m: (
                candidates[m].get("id", "").startswith("spk"),
                candidates[m].get("mean_score", 0.0),
            ))
            kept = candidates[best_idx]
            dropped = [candidates[m].get("id") for m in members if m != best_idx]
            logger.info("  keep %s, drop %s", kept.get("id"), dropped)
            deduped.append(kept)

    return deduped


# ── Main merge ─────────────────────────────────────────────────────────

def merge_candidates() -> SceneSummary:
    """
    Load all upstream outputs and merge into a single SceneSummary.

    Includes qualifying provisional candidates and deduplicates
    candidates whose transcripts are near-identical.
    """
    summary = SceneSummary()

    # ── Load DoA tracks ────────────────────────────────────────────────
    tracks_path = DOA_DIR / "doa_tracks.json"
    if not tracks_path.exists():
        logger.warning("No DoA tracks found – returning empty scene.")
        return summary

    raw = load_json(tracks_path)
    all_candidates: List[Dict[str, Any]] = list(raw.get("candidates", []))

    # ── Include qualifying provisionals ────────────────────────────────
    enh_cfg = CFG.get("enhancement", {})
    if enh_cfg.get("include_provisionals", False):
        min_score = float(enh_cfg.get("provisional_min_score", 0.75))
        min_dur = float(enh_cfg.get("provisional_min_duration_s", 5.0))
        provs = raw.get("provisional_candidates", [])
        accepted = [p for p in provs
                    if p.get("mean_score", 0) >= min_score
                    and p.get("total_duration_s", 0) >= min_dur]
        if accepted:
            logger.info("Including %d/%d provisionals", len(accepted), len(provs))
            all_candidates.extend(accepted)

    logger.info("Total candidates before dedup: %d", len(all_candidates))

    # ── Deduplicate by transcript similarity ───────────────────────────
    deduped = _deduplicate(all_candidates, threshold=0.5)
    logger.info("After deduplication: %d candidates", len(deduped))

    # ── Build SceneSummary ─────────────────────────────────────────────
    for rc in deduped:
        cid = rc.get("id", "spk00")
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
