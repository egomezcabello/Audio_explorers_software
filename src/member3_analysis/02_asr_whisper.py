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


_whisper_model_cache: dict = {}


def _remove_repetitions(text: str, min_phrase_words: int = 4) -> str:
    """
    Remove consecutively repeated phrases of ≥ *min_phrase_words* words.

    Scans for the longest repeating n-gram first (greedy) and keeps
    only the first occurrence.  Iterates until no more repeats found.
    """
    words = text.split()
    changed = True
    while changed:
        changed = False
        n = len(words)
        # Try longest phrases first
        for phrase_len in range(n // 2, min_phrase_words - 1, -1):
            for start in range(n - phrase_len):
                end = start + phrase_len
                if end + phrase_len > n:
                    continue
                phrase = words[start:end]
                # Count consecutive repetitions
                reps = 1
                pos = end
                while pos + phrase_len <= n and words[pos:pos + phrase_len] == phrase:
                    reps += 1
                    pos += phrase_len
                if reps >= 2:
                    # Keep first occurrence, remove the rest
                    words = words[:end] + words[end + phrase_len * (reps - 1):]
                    changed = True
                    break
            if changed:
                break
    return " ".join(words)


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
    """
    from faster_whisper import WhisperModel
    from src.common.paths import WHISPER_MODEL_DIR

    # ── Load model (cached across calls) ──────────────────────────────
    if "model" not in _whisper_model_cache:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        device = "cuda" if has_cuda else "cpu"
        compute_type = "float16" if has_cuda else "float32"
        logger.info("Loading Whisper model '%s' (device=%s, compute=%s)",
                     model_size, device, compute_type)
        _whisper_model_cache["model"] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(WHISPER_MODEL_DIR),
        )
    model = _whisper_model_cache["model"]

    # ── Transcribe ────────────────────────────────────────────────────
    # Force English — all speakers are known to be English
    lang_hint = "en"

    # Read decoding params from config (with safe defaults)
    analysis_cfg = CFG.get("analysis", {})
    beam_size = int(analysis_cfg.get("whisper_beam_size", 5))
    comp_ratio_thr = float(analysis_cfg.get(
        "whisper_compression_ratio_threshold", 2.4))
    log_prob_thr = float(analysis_cfg.get(
        "whisper_log_prob_threshold", -1.0))
    no_speech_thr = float(analysis_cfg.get(
        "whisper_no_speech_threshold", 0.6))
    initial_prompt = analysis_cfg.get("whisper_initial_prompt", None)

    try:
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=beam_size,
            language=lang_hint,
            no_speech_threshold=no_speech_thr,
            log_prob_threshold=log_prob_thr,
            compression_ratio_threshold=comp_ratio_thr,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt=initial_prompt,
        )
        # Materialise the generator and concatenate text
        texts = []
        for seg in segments:
            texts.append(seg.text)
        transcript = " ".join(texts).strip()

        # ── B5: Post-processing repetition removal ────────────────────
        # Detect any phrase of ≥4 words that repeats consecutively
        # and keep only the first occurrence.
        transcript = _remove_repetitions(transcript, min_phrase_words=4)

        logger.info("Whisper: lang=%s prob=%.2f, %d chars",
                     info.language, info.language_probability, len(transcript))
        return transcript

    except Exception as exc:
        logger.error("Whisper transcription failed for %s: %s", audio_path, exc)
        return ""


def _transcribe_subprocess(wav_path: str, out_path: str, model_size: str) -> None:
    """Transcribe a single file in a subprocess (for memory isolation)."""
    import subprocess
    import sys
    import textwrap
    script = textwrap.dedent("""\
        import sys
        sys.path.insert(0, '.')
        from pathlib import Path
        from importlib import import_module
        mod = import_module('src.member3_analysis.02_asr_whisper')
        transcript = mod.transcribe(Path(sys.argv[1]), model_size=sys.argv[2])
        Path(sys.argv[3]).write_text(transcript, encoding='utf-8')
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, wav_path, model_size, out_path],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (rc={result.returncode}): {result.stderr[-500:]}")


def main() -> None:
    """Entry point for step 02 (ASR)."""
    ensure_output_dirs()

    analysis_cfg = CFG.get("analysis", {})
    model_size = analysis_cfg.get("whisper_model", "large-v3")

    wav_files = sorted(SEPARATED_DIR.glob("*_enhanced.wav"))
    if not wav_files:
        logger.warning("No enhanced WAVs found – skipping ASR.")
        return

    for i, wav_path in enumerate(wav_files):
        cid = wav_path.stem.replace("_enhanced", "")
        out_path = ANALYSIS_DIR / f"{cid}_transcript.txt"
        logger.info("Transcribing %s (%d/%d) via subprocess …",
                     wav_path.name, i + 1, len(wav_files))

        try:
            _transcribe_subprocess(str(wav_path), str(out_path), model_size)
            chars = len(out_path.read_text(encoding="utf-8"))
            logger.info("Transcript saved → %s (%d chars)", out_path, chars)
        except Exception as exc:
            logger.error("Transcription failed for %s: %s", wav_path.name, exc)
            out_path.write_text("", encoding="utf-8")

    logger.info("Step 02 (ASR) complete.")


if __name__ == "__main__":
    main()
