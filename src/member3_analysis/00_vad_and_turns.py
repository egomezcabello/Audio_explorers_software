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
    import webrtcvad

    duration = len(audio) / sr
    if duration <= 0:
        return []

    # ── Resample to a rate accepted by webrtcvad ──────────────────────
    TARGET_SR = 16_000
    if sr != TARGET_SR:
        import librosa
        audio_rs = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
        work_sr = TARGET_SR
    else:
        audio_rs = audio
        work_sr = sr

    # Convert to 16-bit PCM
    pcm = (np.clip(audio_rs, -1.0, 1.0) * 32767).astype(np.int16)

    # ── Frame-level VAD decisions ─────────────────────────────────────
    vad = webrtcvad.Vad(mode)
    frame_length = int(work_sr * frame_duration_ms / 1000)  # samples per frame
    frame_bytes = frame_length * 2  # 16-bit = 2 bytes/sample
    n_frames = len(pcm) // frame_length

    is_speech = []
    for i in range(n_frames):
        start = i * frame_length
        chunk = pcm[start : start + frame_length].tobytes()
        if len(chunk) < frame_bytes:
            break
        try:
            is_speech.append(vad.is_speech(chunk, work_sr))
        except Exception:
            is_speech.append(False)

    if not is_speech:
        return []

    # ── Group contiguous speech frames into segments ──────────────────
    frame_dur = frame_duration_ms / 1000.0
    raw_segments: List[Tuple[float, float]] = []
    in_seg = False
    seg_start = 0.0
    for idx, sp in enumerate(is_speech):
        t = idx * frame_dur
        if sp and not in_seg:
            seg_start = t
            in_seg = True
        elif not sp and in_seg:
            raw_segments.append((seg_start, t))
            in_seg = False
    if in_seg:
        raw_segments.append((seg_start, min(n_frames * frame_dur, duration)))

    if not raw_segments:
        return []

    # ── Merge segments separated by small gaps ────────────────────────
    GAP_MERGE = 0.10  # seconds
    merged: List[Tuple[float, float]] = [raw_segments[0]]
    for s, e in raw_segments[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= GAP_MERGE:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    # ── Remove segments shorter than min_speech_duration_s ────────────
    min_dur = CFG.get("analysis", {}).get("min_speech_duration_s", 0.3)
    segments = [(s, e) for s, e in merged if (e - s) >= min_dur]

    logger.info("VAD: %d raw → %d merged → %d final segments (%.1f s speech / %.1f s total)",
                len(raw_segments), len(merged), len(segments),
                sum(e - s for s, e in segments), duration)
    return segments


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
