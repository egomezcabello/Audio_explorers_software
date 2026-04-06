#!/usr/bin/env python3
"""
00_vad_and_turns.py – Voice Activity Detection and turn segmentation.

Loads each enhanced candidate WAV from ``outputs/separated/`` and runs
VAD to find speech / silence boundaries.

TODO:
    - Integrate ``webrtcvad`` or an energy-based VAD.
    - Segment speech into turns (list of [start_s, end_s]).
    - Filter out very short segments (< min_speech_duration_s).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.common.config import CFG
from src.common.constants import SAMPLE_RATE
from src.common.logging_utils import setup_logging
from src.common.paths import ANALYSIS_DIR, SEPARATED_DIR, ensure_output_dirs

logger = setup_logging(__name__)


def run_vad(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    mode: int = 3,
    frame_duration_ms: int = 30,
) -> List[Tuple[float, float]]:
    """
    Run Voice Activity Detection on a mono signal.

    Parameters
    ----------
    audio : np.ndarray
        1-D waveform (float, ±1 range).
    sr : int
        Sample rate.
    mode : int
        webrtcvad aggressiveness (0 = least aggressive, 3 = most).
    frame_duration_ms : int
        Frame size for VAD decisions.

    Returns
    -------
    segments : list[tuple[float, float]]
        List of ``(start_seconds, end_seconds)`` for each speech segment.

    TODO
    ----
    - Implement using ``webrtcvad``.
    - Merge adjacent segments with small gaps.
    - Filter by minimum duration.
    """
    # TODO: Implement VAD
    logger.warning("run_vad() is a placeholder – returning full duration as speech.")
    duration = len(audio) / sr
    if duration > 0:
        return [(0.0, duration)]
    return []


def save_vad_results(
    candidate_id: str,
    segments: List[Tuple[float, float]],
    out_dir: Path,
) -> Path:
    """Save VAD segments as a NumPy file for later steps."""
    out_path = out_dir / f"{candidate_id}_vad.npy"
    np.save(str(out_path), np.array(segments, dtype=np.float64))
    logger.info("VAD for %s: %d segment(s) → %s", candidate_id, len(segments), out_path)
    return out_path


def main() -> None:
    """Entry point for step 00 (VAD)."""
    ensure_output_dirs()

    analysis_cfg = CFG.get("analysis", {})
    vad_mode = analysis_cfg.get("vad_mode", 3)

    # Find all enhanced WAV files
    wav_files = sorted(SEPARATED_DIR.glob("*_enhanced.wav"))
    if not wav_files:
        logger.warning("No enhanced WAVs found in %s – nothing to process.", SEPARATED_DIR)

    for wav_path in wav_files:
        cid = wav_path.stem.replace("_enhanced", "")
        logger.info("Running VAD on %s …", wav_path.name)

        import soundfile as sf
        audio, sr = sf.read(str(wav_path), dtype="float64")

        segments = run_vad(audio, sr=sr, mode=vad_mode)
        save_vad_results(cid, segments, ANALYSIS_DIR)

    logger.info("Step 00 (VAD) complete.")


if __name__ == "__main__":
    main()
