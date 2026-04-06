#!/usr/bin/env python3
"""
04_pack_results.py – Package all per-candidate analysis into JSON.

Collects VAD, Language ID, ASR transcript, and pitch results for each
candidate and writes a single ``spkXX_analysis.json``.

Output schema per candidate:
{
    "candidate_id": "spk00",
    "vad_segments": [[start_s, end_s], ...],
    "language": "en",
    "language_confidence": 0.95,
    "transcript": "Hello world …",
    "pitch": {
        "median_f0_hz": 130.5,
        "voice_type": "male"
    }
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.common.logging_utils import setup_logging
from src.common.paths import ANALYSIS_DIR, SEPARATED_DIR, ensure_output_dirs

logger = setup_logging(__name__)


def pack_candidate(
    cid: str,
    analysis_dir: Path,
) -> Dict[str, Any]:
    """
    Gather all analysis artefacts for one candidate into a single dict.

    Parameters
    ----------
    cid : str
        Candidate identifier (e.g. ``"spk00"``).
    analysis_dir : Path
        Directory containing intermediate analysis files.

    Returns
    -------
    dict
        Merged analysis record.
    """
    record: Dict[str, Any] = {"candidate_id": cid}

    # VAD segments
    vad_path = analysis_dir / f"{cid}_vad.npy"
    if vad_path.exists():
        segs = np.load(str(vad_path)).tolist()
        record["vad_segments"] = segs
    else:
        record["vad_segments"] = []

    # Language ID
    lang_path = analysis_dir / "language_id_results.json"
    if lang_path.exists():
        with open(lang_path, "r", encoding="utf-8") as fh:
            lang_data = json.load(fh)
        cand_lang = lang_data.get(cid, {})
        record["language"] = cand_lang.get("language", "unknown")
        record["language_confidence"] = cand_lang.get("confidence", 0.0)
    else:
        record["language"] = "unknown"
        record["language_confidence"] = 0.0

    # Transcript
    tx_path = analysis_dir / f"{cid}_transcript.txt"
    if tx_path.exists():
        record["transcript"] = tx_path.read_text(encoding="utf-8")
    else:
        record["transcript"] = ""

    # Pitch
    pitch_path = analysis_dir / "pitch_results.json"
    if pitch_path.exists():
        with open(pitch_path, "r", encoding="utf-8") as fh:
            pitch_data = json.load(fh)
        for pr in pitch_data:
            if pr.get("candidate_id") == cid:
                record["pitch"] = {
                    "median_f0_hz": pr.get("median_f0_hz", 0.0),
                    "std_f0_hz": pr.get("std_f0_hz", 0.0),
                    "voice_type": pr.get("voice_type", "unknown"),
                }
                break
        else:
            record["pitch"] = {}
    else:
        record["pitch"] = {}

    return record


def main() -> None:
    """Entry point for step 04 (pack results)."""
    ensure_output_dirs()

    wav_files = sorted(SEPARATED_DIR.glob("*_enhanced.wav"))
    candidate_ids = [p.stem.replace("_enhanced", "") for p in wav_files]

    if not candidate_ids:
        logger.warning("No candidates found – nothing to pack.")
        return

    for cid in candidate_ids:
        record = pack_candidate(cid, ANALYSIS_DIR)
        out_path = ANALYSIS_DIR / f"{cid}_analysis.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)
        logger.info("Packed analysis → %s", out_path)

    logger.info("Step 04 (pack) complete.")


if __name__ == "__main__":
    main()
