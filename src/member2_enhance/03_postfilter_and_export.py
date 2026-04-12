#!/usr/bin/env python3
"""
03_postfilter_and_export.py – Post-filter enhanced STFTs and export WAVs.
==========================================================================
Fourth (final) step of the Member 2 (Enhancement) pipeline.

Applies optional post-filtering to each candidate's beamformed STFT,
then converts back to time domain and exports per-candidate WAV files.

Post-filter methods
-------------------
- ``"none"``:  no filtering — pass through unchanged.
- ``"wiener"``:  conservative Wiener-style gain using a smoothed power
  envelope and a noise-floor estimate from the quietest frames of the
  enhanced signal itself.
- ``"binary_mask"``:  a gentle binary mask derived from the same Wiener
  gain estimate — keeps bins above a threshold, attenuates the rest.

STFT convention: ``(n_freq, n_frames)`` for the enhanced mono STFT.

Outputs
-------
- ``outputs/separated/spkXX_enhanced.wav``
- ``outputs/separated/spkXX_debug.npz``  (updated with final waveform)
"""

from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import uniform_filter

from src.common.config import CFG
from src.common.constants import SAMPLE_RATE
from src.common.io_utils import save_mono_wav
from src.common.logging_utils import setup_logging
from src.common.paths import DOA_DIR, SEPARATED_DIR, ensure_output_dirs
from src.common.stft_utils import compute_istft

logger = setup_logging(__name__)


# ── Post-filter implementations ───────────────────────────────────────

def _estimate_noise_floor(power: np.ndarray, quantile: float = 0.10) -> np.ndarray:
    """
    Estimate a per-frequency noise floor from the quietest frames.

    Parameters
    ----------
    power : np.ndarray, shape (n_freq, n_frames)
    quantile : float
        Fraction of quietest frames used to estimate noise.

    Returns
    -------
    noise_floor : np.ndarray, shape (n_freq, 1)
    """
    n_frames = power.shape[1]
    k = max(1, int(quantile * n_frames))
    sorted_power = np.sort(power, axis=1)
    noise_floor = sorted_power[:, :k].mean(axis=1, keepdims=True)
    return noise_floor


def apply_postfilter(
    enhanced_stft: np.ndarray,
    method: str = "none",
) -> np.ndarray:
    """
    Apply a post-filter to the beamformed STFT.

    Parameters
    ----------
    enhanced_stft : np.ndarray, shape (n_freq, n_frames) — complex
    method : str
        ``"none"`` | ``"wiener"`` | ``"binary_mask"``

    Returns
    -------
    np.ndarray
        Post-filtered STFT, same shape.
    """
    if method == "none":
        return enhanced_stft

    power = np.abs(enhanced_stft) ** 2
    # Smooth power in time (window of 5 frames) for stability
    smooth_power = uniform_filter(power, size=(1, 5), mode="reflect")
    noise_floor = _estimate_noise_floor(power, quantile=0.10)

    # Wiener-style gain: signal / (signal + noise)
    gain = smooth_power / (smooth_power + noise_floor + 1e-12)
    # Clip to be conservative — never boost, only attenuate
    gain = np.clip(gain, 0.0, 1.0)

    if method == "wiener":
        # Mild square-root gain for gentler attenuation
        gain = np.sqrt(gain)
        return enhanced_stft * gain

    if method == "binary_mask":
        # Threshold the Wiener gain at 0.5 — but soften the transition
        # to avoid harsh artefacts
        threshold = 0.5
        binary = np.where(gain >= threshold, 1.0, 0.1)
        return enhanced_stft * binary

    logger.warning("[member2][export] unknown postfilter '%s' — skipping.",
                   method)
    return enhanced_stft


# ── Entry point ────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for step 03 (post-filter & export)."""
    ensure_output_dirs()

    enh_cfg = CFG.get("enhancement", {})
    postfilter = enh_cfg.get("postfilter", "none")

    # ── Load candidate list ────────────────────────────────────────────
    tracks_path = DOA_DIR / "doa_tracks.json"
    if not tracks_path.exists():
        raise FileNotFoundError(
            f"[member2][export] DoA tracks not found: {tracks_path}"
        )
    with open(tracks_path, "r", encoding="utf-8") as fh:
        scene = json.load(fh)
    candidates = scene.get("candidates", [])

    # ── Optionally include provisional candidates ──────────────────
    enh_cfg2 = CFG.get("enhancement", {})
    if enh_cfg2.get("include_provisionals", False):
        min_score = float(enh_cfg2.get("provisional_min_score", 0.75))
        min_dur = float(enh_cfg2.get("provisional_min_duration_s", 5.0))
        provs = scene.get("provisional_candidates", [])
        accepted = [p for p in provs
                    if p.get("mean_score", 0) >= min_score
                    and p.get("total_duration_s", 0) >= min_dur]
        if accepted:
            logger.info("[member2][export] including %d/%d provisionals",
                        len(accepted), len(provs))
            candidates = candidates + accepted

    if not candidates:
        logger.warning("[member2][export] no candidates — nothing to export.")
        return
    logger.info("[member2][export] %d candidate(s), postfilter='%s'",
                len(candidates), postfilter)

    # ── Per-candidate export ───────────────────────────────────────────
    for cand in candidates:
        cid = cand.get("id", "spk00")
        stft_path = SEPARATED_DIR / f"{cid}_enhanced_stft.npy"

        if not stft_path.exists():
            logger.warning("[member2][export] %s enhanced STFT not found — "
                           "skipping.", cid)
            continue

        enhanced_stft = np.load(str(stft_path))

        # Post-filter
        filtered = apply_postfilter(enhanced_stft, method=postfilter)

        # iSTFT → waveform
        _, audio = compute_istft(filtered)

        # Gentle normalisation only if clipping would occur
        peak = np.abs(audio).max()
        if peak > 0.99:
            audio = audio * (0.95 / peak)

        # Save WAV
        wav_path = SEPARATED_DIR / f"{cid}_enhanced.wav"
        save_mono_wav(wav_path, audio, sr=SAMPLE_RATE)

        # Update debug file with final outputs
        debug_path = SEPARATED_DIR / f"{cid}_debug.npz"
        debug_data = {}
        if debug_path.exists():
            with np.load(str(debug_path), allow_pickle=True) as d:
                debug_data = dict(d)
        debug_data["filtered_stft"] = filtered
        debug_data["waveform"] = audio
        debug_data["postfilter_method"] = np.array(postfilter)
        np.savez_compressed(str(debug_path), **debug_data)

        dur_s = len(audio) / SAMPLE_RATE
        logger.info("[member2][export] %s: %.2fs  peak=%.3f  → %s",
                    cid, dur_s, peak, wav_path.name)

    logger.info("[member2][export] done — %d WAV(s) exported",
                len(candidates))


if __name__ == "__main__":
    main()
