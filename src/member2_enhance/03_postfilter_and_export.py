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
from src.common.paths import DOA_DIR, INTERMEDIATE_DIR, SEPARATED_DIR, ensure_output_dirs
from src.common.stft_utils import compute_istft

logger = setup_logging(__name__)


# ── Spatial mask gating ────────────────────────────────────────────────

def apply_spatial_mask_gate(
    enhanced_stft: np.ndarray,
    mask: np.ndarray,
    floor: float = 0.05,
    exponent: float = 1.0,
) -> np.ndarray:
    """
    Gate the enhanced STFT using the spatial mask from mask-building.

    T-F bins where the target mask is low (= interferer-dominated)
    are attenuated.  This suppresses crosstalk frames that the MVDR
    beamformer cannot reject because the interferer is too close
    angularly (e.g. 45° apart on a 4-mic BTE array).

    Parameters
    ----------
    enhanced_stft : (n_freq, n_frames) complex
    mask : (n_freq, n_frames_mask) float  — spatial mask [0, 1]
    floor : float
        Minimum gain (prevents total zeroing / artefacts).
    exponent : float
        Sharpen the gate by raising mask to this power.

    Returns
    -------
    gated : (n_freq, n_frames) complex
    """
    n_freq, n_frames = enhanced_stft.shape
    n_freq_m, n_frames_m = mask.shape

    # Align mask to enhanced STFT dimensions if needed
    if n_frames_m != n_frames:
        from scipy.ndimage import zoom
        scale = (n_freq / n_freq_m, n_frames / n_frames_m)
        mask = zoom(mask, scale, order=1)
    elif n_freq_m != n_freq:
        from scipy.ndimage import zoom
        scale = (n_freq / n_freq_m, 1.0)
        mask = zoom(mask, scale, order=1)

    gate = np.clip(mask, 0.0, 1.0)
    if exponent != 1.0:
        gate = np.power(gate, exponent)
    gate = np.clip(gate, floor, 1.0)

    return enhanced_stft * gate


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

    # ── Clean stale outputs from previous runs ──────────────────────────
    for stale in list(SEPARATED_DIR.glob("*_enhanced.wav")) + \
                 list(SEPARATED_DIR.glob("*_debug.npz")):
        stale.unlink()
    logger.info("[member2][export] cleaned stale WAV/debug files in %s", SEPARATED_DIR)

    enh_cfg = CFG.get("enhancement", {})
    postfilter = enh_cfg.get("postfilter", "none")
    use_mask_gate = enh_cfg.get("use_spatial_mask_gate", False)
    mask_gate_floor = float(enh_cfg.get("mask_gate_floor", 0.05))
    mask_gate_exponent = float(enh_cfg.get("mask_gate_exponent", 1.0))

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
        min_sep = float(enh_cfg2.get("provisional_min_sep_deg", 0.0))
        provs = scene.get("provisional_candidates", [])
        conf_azs = [c.get("mean_azimuth", 0) for c in candidates]
        accepted = []
        for p in provs:
            if p.get("mean_score", 0) < min_score:
                continue
            if p.get("total_duration_s", 0) < min_dur:
                continue
            if min_sep > 0 and conf_azs:
                p_az = p.get("mean_azimuth", 0)
                too_close = any(
                    min(abs(p_az - ca), 360 - abs(p_az - ca)) < min_sep
                    for ca in conf_azs
                )
                if too_close:
                    continue
            accepted.append(p)
        if accepted:
            logger.info("[member2][export] including %d/%d provisionals",
                        len(accepted), len(provs))
            candidates = candidates + accepted

    # ── Honour step-01 dedup manifest if available ──────────────────
    manifest_path = INTERMEDIATE_DIR / "active_candidates.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as _mf:
            manifest = json.load(_mf)
        if isinstance(manifest, dict):
            active_ids = set(manifest.get("active", []))
        else:
            active_ids = set(manifest)
        candidates = [c for c in candidates if c.get("id") in active_ids]

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

        # Spatial mask gating — suppress T-F bins where target mask is low
        if use_mask_gate:
            mask_path = INTERMEDIATE_DIR / f"{cid}_mask.npy"
            if mask_path.exists():
                mask = np.load(str(mask_path)).astype(np.float64)
                pre_power = float(np.mean(np.abs(enhanced_stft) ** 2))
                enhanced_stft = apply_spatial_mask_gate(
                    enhanced_stft, mask,
                    floor=mask_gate_floor,
                    exponent=mask_gate_exponent,
                )
                post_power = float(np.mean(np.abs(enhanced_stft) ** 2))
                ratio = post_power / (pre_power + 1e-12)
                logger.info("[member2][export] %s: mask gate applied "
                            "(floor=%.2f, exp=%.1f, power_ratio=%.3f)",
                            cid, mask_gate_floor, mask_gate_exponent, ratio)
            else:
                logger.warning("[member2][export] %s: mask %s not found "
                               "— skipping gate", cid, mask_path.name)

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

        # Update debug file with final outputs (gated)
        save_debug = CFG.get("pipeline", {}).get("save_debug", False)
        if save_debug:
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
