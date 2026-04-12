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
    _model_cache: dict = {},
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
    """
    from faster_whisper import WhisperModel
    from src.common.paths import WHISPER_MODEL_DIR

    # ── Load model (cached across calls) ──────────────────────────────
    if "model" not in _model_cache:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        device = "cuda" if has_cuda else "cpu"
        compute_type = "float16" if has_cuda else "float32"
        logger.info("Loading Whisper model '%s' (device=%s, compute=%s)",
                     model_size, device, compute_type)
        _model_cache["model"] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(WHISPER_MODEL_DIR),
        )
    model = _model_cache["model"]

    # ── Transcribe ────────────────────────────────────────────────────
    # Force English — all speakers are known to be English
    lang_hint = "en"

    try:
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            language=lang_hint,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        # Materialise the generator and concatenate text
        texts = []
        for seg in segments:
            texts.append(seg.text)
        transcript = " ".join(texts).strip()

        logger.info("Whisper: lang=%s prob=%.2f, %d chars",
                     info.language, info.language_probability, len(transcript))
        return transcript

    except Exception as exc:
        logger.error("Whisper transcription failed for %s: %s", audio_path, exc)
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

        # Free memory between files to avoid OOM with large models
        import gc
        gc.collect()

    logger.info("Step 02 (ASR) complete.")


if __name__ == "__main__":
    main()
