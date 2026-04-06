#!/usr/bin/env python3
"""
03_pitch_voice_type.py – Pitch estimation and voice-type classification.

Estimates the fundamental frequency (F0) contour for each candidate and
classifies voice type (e.g., male / female / child).

TODO:
    - Implement F0 estimation (e.g., pYIN via librosa, or CREPE).
    - Compute pitch statistics (median, std, range).
    - Classify voice type from pitch stats.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.common.constants import SAMPLE_RATE
from src.common.logging_utils import setup_logging
from src.common.paths import ANALYSIS_DIR, SEPARATED_DIR, ensure_output_dirs

logger = setup_logging(__name__)


@dataclass
class PitchResult:
    """Holds pitch analysis for one candidate."""
    candidate_id: str = ""
    median_f0_hz: float = 0.0
    std_f0_hz: float = 0.0
    min_f0_hz: float = 0.0
    max_f0_hz: float = 0.0
    voice_type: str = "unknown"  # "male", "female", "child", "unknown"


def estimate_pitch(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Estimate the F0 contour of a speech signal.

    Parameters
    ----------
    audio : np.ndarray
        1-D waveform.
    sr : int
        Sample rate.

    Returns
    -------
    f0 : np.ndarray
        Frame-wise F0 in Hz (0 for unvoiced frames).

    TODO
    ----
    - Implement using ``librosa.pyin`` or similar.
    """
    # TODO: Implement pitch estimation
    logger.warning("estimate_pitch() is a placeholder – returning zeros.")
    n_frames = max(1, len(audio) // 512)
    return np.zeros(n_frames, dtype=np.float64)


def classify_voice_type(median_f0: float) -> str:
    """
    Simple heuristic voice-type classification from median F0.

    Returns ``"male"`` / ``"female"`` / ``"child"`` / ``"unknown"``.

    TODO
    ----
    - Refine thresholds with real data.
    - Consider using a trained classifier instead.
    """
    if median_f0 <= 0:
        return "unknown"
    if median_f0 < 165:
        return "male"
    if median_f0 < 255:
        return "female"
    return "child"


def main() -> None:
    """Entry point for step 03 (pitch)."""
    ensure_output_dirs()

    wav_files = sorted(SEPARATED_DIR.glob("*_enhanced.wav"))
    if not wav_files:
        logger.warning("No enhanced WAVs found – skipping pitch analysis.")
        return

    results = []

    for wav_path in wav_files:
        cid = wav_path.stem.replace("_enhanced", "")
        logger.info("Pitch analysis on %s …", wav_path.name)

        import soundfile as sf
        audio, sr = sf.read(str(wav_path), dtype="float64")

        f0 = estimate_pitch(audio, sr)
        voiced = f0[f0 > 0]

        pr = PitchResult(candidate_id=cid)
        if len(voiced) > 0:
            pr.median_f0_hz = float(np.median(voiced))
            pr.std_f0_hz = float(np.std(voiced))
            pr.min_f0_hz = float(np.min(voiced))
            pr.max_f0_hz = float(np.max(voiced))
        pr.voice_type = classify_voice_type(pr.median_f0_hz)

        results.append(pr)
        logger.info("  → median F0=%.1f Hz, type=%s", pr.median_f0_hz, pr.voice_type)

    # Save combined pitch results
    import json
    from dataclasses import asdict
    out_path = ANALYSIS_DIR / "pitch_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)
    logger.info("Pitch results saved → %s", out_path)
    logger.info("Step 03 (pitch) complete.")


if __name__ == "__main__":
    main()
