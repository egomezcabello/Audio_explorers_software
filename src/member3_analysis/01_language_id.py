#!/usr/bin/env python3
"""
01_language_id.py – Language identification per candidate.

Uses SpeechBrain's pre-trained language-ID model to identify the spoken
language in each enhanced candidate WAV.

TODO:
    - Load the SpeechBrain LangID model (VoxLingua107-ECAPA).
    - Run inference on each candidate's speech segments.
    - Store language ID + confidence for each candidate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.common.config import CFG
from src.common.logging_utils import setup_logging
from src.common.paths import ANALYSIS_DIR, SEPARATED_DIR, ensure_output_dirs

logger = setup_logging(__name__)


def identify_language(
    audio: np.ndarray,
    sr: int,
    model_name: str = "speechbrain/lang-id-voxlingua107-ecapa",
) -> Tuple[str, float]:
    """
    Identify the language of a speech signal.

    Parameters
    ----------
    audio : np.ndarray
        1-D waveform.
    sr : int
        Sample rate.
    model_name : str
        SpeechBrain model identifier.

    Returns
    -------
    language : str
        ISO 639-1 language code (e.g. ``"en"``).
    confidence : float
        Posterior probability for the predicted language.

    TODO
    ----
    - Load SpeechBrain model (cache in models/speechbrain/).
    - Run inference.
    - Return top-1 language and confidence.
    """
    # TODO: Implement language ID
    logger.warning("identify_language() is a placeholder – returning 'unknown'.")
    return "unknown", 0.0


def main() -> None:
    """Entry point for step 01 (Language ID)."""
    ensure_output_dirs()

    analysis_cfg = CFG.get("analysis", {})
    model_name = analysis_cfg.get(
        "language_id_model", "speechbrain/lang-id-voxlingua107-ecapa"
    )

    wav_files = sorted(SEPARATED_DIR.glob("*_enhanced.wav"))
    if not wav_files:
        logger.warning("No enhanced WAVs found – skipping Language ID.")
        return

    results: Dict[str, Dict] = {}

    for wav_path in wav_files:
        cid = wav_path.stem.replace("_enhanced", "")
        logger.info("Language ID on %s …", wav_path.name)

        import soundfile as sf
        audio, sr = sf.read(str(wav_path), dtype="float64")

        lang, conf = identify_language(audio, sr, model_name=model_name)
        results[cid] = {"language": lang, "confidence": round(conf, 4)}
        logger.info("  → %s (%.2f%%)", lang, conf * 100)

    # Save combined results
    import json
    out_path = ANALYSIS_DIR / "language_id_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Language ID results saved → %s", out_path)
    logger.info("Step 01 (Language ID) complete.")


if __name__ == "__main__":
    main()
