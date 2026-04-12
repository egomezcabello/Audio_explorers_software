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
    """Score DoA stability – low circular std → high score."""
    if not doa_track or len(doa_track) < 2:
        return 0.0
    azimuths = [e[1] for e in doa_track if len(e) >= 2]
    if len(azimuths) < 2:
        return 0.0
    az_rad = np.deg2rad(azimuths)
    R = np.sqrt(np.mean(np.cos(az_rad)) ** 2 + np.mean(np.sin(az_rad)) ** 2)
    circ_std_deg = np.rad2deg(np.sqrt(-2 * np.log(max(R, 1e-6))))
    # 0° std → 1.0, 30° std → 0.0
    return float(np.clip(1.0 - circ_std_deg / 30.0, 0.0, 1.0))


def score_speech_duration(analysis: Dict[str, Any]) -> float:
    """Score based on total speech duration from VAD segments."""
    segments = analysis.get("vad_segments", [])
    if not segments:
        return 0.0
    total = sum(seg[1] - seg[0] for seg in segments if len(seg) >= 2)
    # Normalise: 15 s of speech in a 21 s clip → 1.0
    return float(np.clip(total / 15.0, 0.0, 1.0))


def score_language_match(analysis: Dict[str, Any], target_lang: str = "en") -> float:
    """Score based on language match to expected target."""
    lang = analysis.get("language", "unknown")
    conf = float(analysis.get("language_confidence", 0.0))
    return conf if lang == target_lang else 0.0


def score_snr(candidate_id: str) -> float:
    """Estimate SNR from the enhanced signal's energy distribution."""
    wav_path = SEPARATED_DIR / f"{candidate_id}_enhanced.wav"
    if not wav_path.exists():
        return 0.0
    try:
        import soundfile as sf
        audio, sr_ = sf.read(str(wav_path), dtype="float64")
        frame_len = int(0.025 * sr_)  # 25 ms frames
        n_frames = len(audio) // frame_len
        if n_frames < 5:
            return 0.5
        energies = np.array([
            np.mean(audio[i * frame_len:(i + 1) * frame_len] ** 2)
            for i in range(n_frames)
        ])
        energies.sort()
        noise_est = np.mean(energies[:max(1, n_frames // 5)])   # bottom 20 %
        signal_est = np.mean(energies[n_frames // 2:])           # top 50 %
        if noise_est < 1e-12:
            return 1.0
        snr_db = 10 * np.log10(signal_est / noise_est + 1e-12)
        # 0 dB → 0.0, 30 dB → 1.0
        return float(np.clip(snr_db / 30.0, 0.0, 1.0))
    except Exception:
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
