#!/usr/bin/env python3
"""
03_postfilter_and_export.py – Post-filter enhanced STFTs and export WAVs.

Applies optional post-filtering (Wiener, binary mask, …) to the MVDR
output, then converts back to time domain and saves per-candidate WAV
files to ``outputs/separated/``.

TODO:
    - Implement Wiener or binary post-filter.
    - Convert STFT back to waveform via iSTFT.
    - Normalise and save each candidate as WAV.
    - Optionally save debug NPZ with intermediate arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.common.config import CFG
from src.common.constants import SAMPLE_RATE
from src.common.io_utils import save_mono_wav
from src.common.json_schema import load_json
from src.common.logging_utils import setup_logging
from src.common.paths import DOA_DIR, SEPARATED_DIR, ensure_output_dirs
from src.common.stft_utils import compute_istft

logger = setup_logging(__name__)


def apply_postfilter(
    enhanced_stft: np.ndarray,
    method: str = "none",
) -> np.ndarray:
    """
    Apply a post-filter to the beamformed STFT.

    Parameters
    ----------
    enhanced_stft : np.ndarray
        Shape ``(n_freq, n_frames)`` – complex.
    method : str
        ``"none"`` | ``"wiener"`` | ``"binary_mask"``.

    Returns
    -------
    np.ndarray
        Post-filtered STFT, same shape.

    TODO
    ----
    - Implement Wiener post-filter.
    - Implement binary mask post-filter.
    """
    if method == "none":
        return enhanced_stft

    # TODO: Implement post-filters
    logger.warning("apply_postfilter('%s') is a placeholder – returning input.", method)
    return enhanced_stft


def main() -> None:
    """Entry point for step 03 (post-filter & export)."""
    ensure_output_dirs()

    enh_cfg = CFG.get("enhancement", {})
    postfilter = enh_cfg.get("postfilter", "none")

    # Load candidates list
    tracks_path = DOA_DIR / "doa_tracks.json"
    if tracks_path.exists():
        scene = load_json(tracks_path)
        candidates = scene.get("candidates", [])
    else:
        candidates = []

    for cand in candidates:
        cid = cand.get("id", "spk00")
        stft_path = SEPARATED_DIR / f"{cid}_enhanced_stft.npy"

        if stft_path.exists():
            enhanced_stft = np.load(str(stft_path))
        else:
            logger.warning("Enhanced STFT for %s not found – generating silence.", cid)
            enhanced_stft = np.zeros((513, 100), dtype=np.complex64)

        # Post-filter
        filtered = apply_postfilter(enhanced_stft, method=postfilter)

        # iSTFT → waveform
        _, audio = compute_istft(filtered)

        # Save WAV
        wav_path = SEPARATED_DIR / f"{cid}_enhanced.wav"
        save_mono_wav(wav_path, audio, sr=SAMPLE_RATE)

        # Optionally save debug info
        debug_path = SEPARATED_DIR / f"{cid}_debug.npz"
        np.savez_compressed(
            str(debug_path),
            enhanced_stft=filtered,
            waveform=audio,
        )
        logger.info("Exported %s → %s + %s", cid, wav_path.name, debug_path.name)

    logger.info("Step 03 (export) complete.")


if __name__ == "__main__":
    main()
