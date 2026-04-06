#!/usr/bin/env python3
"""
02_asr_whisper.py – Automatic Speech Recognition using faster-whisper.

Transcribes each enhanced candidate WAV and saves the output to
``outputs/analysis/spkXX_transcript.txt``.

TODO:
    - Load faster-whisper model (cache in models/whisper/).
    - Transcribe each candidate's WAV.
    - Handle language hint from Language ID step.
    - Save word-level timestamps if available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.common.config import CFG
from src.common.logging_utils import setup_logging
from src.common.paths import ANALYSIS_DIR, SEPARATED_DIR, ensure_output_dirs

logger = setup_logging(__name__)


def transcribe(
    audio_path: Path,
    model_size: str = "large-v3",
    language: Optional[str] = None,
) -> str:
    """
    Transcribe a WAV file using faster-whisper.

    Parameters
    ----------
    audio_path : Path
        Path to mono WAV.
    model_size : str
        Whisper model size (``"tiny"``, ``"base"``, …, ``"large-v3"``).
    language : str or None
        Optional language hint (ISO 639-1 code).

    Returns
    -------
    transcript : str
        Full transcription text.

    TODO
    ----
    - Load faster_whisper.WhisperModel (with device/compute_type config).
    - Run model.transcribe() and collect segments.
    - Concatenate segment texts.
    """
    # TODO: Implement Whisper transcription
    logger.warning("transcribe() is a placeholder – returning empty string.")
    return ""


def main() -> None:
    """Entry point for step 02 (ASR)."""
    ensure_output_dirs()

    analysis_cfg = CFG.get("analysis", {})
    model_size = analysis_cfg.get("whisper_model", "large-v3")

    wav_files = sorted(SEPARATED_DIR.glob("*_enhanced.wav"))
    if not wav_files:
        logger.warning("No enhanced WAVs found – skipping ASR.")
        return

    # Optionally load language hints from earlier step
    lang_path = ANALYSIS_DIR / "language_id_results.json"
    lang_hints = {}
    if lang_path.exists():
        import json
        with open(lang_path, "r", encoding="utf-8") as fh:
            lang_hints = json.load(fh)

    for wav_path in wav_files:
        cid = wav_path.stem.replace("_enhanced", "")
        lang = lang_hints.get(cid, {}).get("language", None)
        logger.info("Transcribing %s (lang hint=%s) …", wav_path.name, lang)

        transcript = transcribe(wav_path, model_size=model_size, language=lang)

        out_path = ANALYSIS_DIR / f"{cid}_transcript.txt"
        out_path.write_text(transcript, encoding="utf-8")
        logger.info("Transcript saved → %s (%d chars)", out_path, len(transcript))

    logger.info("Step 02 (ASR) complete.")


if __name__ == "__main__":
    main()
