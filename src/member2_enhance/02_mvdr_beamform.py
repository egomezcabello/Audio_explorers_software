#!/usr/bin/env python3
"""
02_mvdr_beamform.py – MVDR beamforming per candidate.
======================================================
Third step of the Member 2 (Enhancement) pipeline.

Supports two steering modes (``enhancement.steering_mode``):

- ``"doa_model"``: build steering vector from Member 1 calibration
  delay model + candidate mean azimuth.  More direction-specific.
- ``"eigen"``: extract steering vector from the principal eigenvector
  of the target covariance matrix (original method).

STFT convention: ``(n_channels, n_freq, n_frames)``.

Outputs
-------
- ``outputs/separated/spkXX_enhanced_stft.npy``
- ``outputs/separated/spkXX_debug.npz``
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

import numpy as np

from src.common.config import CFG, get_channel_order, get_stft_params
from src.common.constants import SAMPLE_RATE
from src.common.logging_utils import setup_logging
from src.common.paths import (
    CALIB_DIR, DOA_DIR, INTERMEDIATE_DIR, SEPARATED_DIR, ensure_output_dirs,
)

logger = setup_logging(__name__)


# ── Covariance estimation ─────────────────────────────────────────────

def estimate_covariance(
    stft: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Estimate a spatial covariance matrix from a masked STFT.

    Parameters
    ----------
    stft : shape (n_ch, n_freq, n_frames)
    mask : shape (n_freq, n_frames) — values in [0, 1]

    Returns
    -------
    cov : shape (n_freq, n_ch, n_ch) — complex
    """
    n_ch, n_freq, n_frames = stft.shape
    cov = np.zeros((n_freq, n_ch, n_ch), dtype=np.complex128)

    for f in range(n_freq):
        x = stft[:, f, :]
        m = mask[f, :]
        x_weighted = x * m[np.newaxis, :]
        cov[f] = (x_weighted @ x.conj().T) / (m.sum() + eps)

    return cov


# ── Steering: eigenvector method (original) ───────────────────────────

def compute_steering_eigen(
    cov_target: np.ndarray,
) -> np.ndarray:
    """
    Extract steering vector as principal eigenvector of target covariance.
    Phase-normalised so channel 0 has real positive phase.

    Returns shape (n_freq, n_ch).
    """
    n_freq, n_ch, _ = cov_target.shape
    steer = np.zeros((n_freq, n_ch), dtype=np.complex128)

    for f in range(n_freq):
        eigvals, eigvecs = np.linalg.eigh(cov_target[f])
        d = eigvecs[:, -1]
        d = d * np.exp(-1j * np.angle(d[0]))
        d = d / (np.linalg.norm(d) + 1e-12)
        steer[f] = d

    return steer


# ── Steering: DoA delay-model method (new) ────────────────────────────

def compute_steering_doa_model(
    mean_azimuth_deg: float,
    calibration: dict,
    n_freq: int,
    n_fft: int,
    sr: int,
    channel_order: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build steering vector from calibration delay model at given azimuth.

    The delay model gives pairwise TDOA as:
        tau_pair(theta) = a * cos(theta) + b * sin(theta)

    We recover per-mic absolute delays via least squares (fixing LF = 0),
    then build:
        d[f, m] = exp(-j * 2*pi * freq[f] * t_m)

    Returns
    -------
    steer : shape (n_freq, n_ch)
    mic_delays : shape (n_ch,) — relative delays in seconds
    """
    delay_model = calibration["delay_model"]
    theta_rad = np.deg2rad(mean_azimuth_deg)

    # Predict pairwise delays at this azimuth
    pair_taus = {}
    for pair_name, model in delay_model.items():
        a, b = model["a"], model["b"]
        pair_taus[pair_name] = a * np.cos(theta_rad) + b * np.sin(theta_rad)

    # Solve for per-mic delays via least squares.
    # Fix reference channel (index 0) to delay 0.
    n_ch = len(channel_order)
    ref_ch = channel_order[0]
    non_ref = [ch for ch in channel_order if ch != ref_ch]
    non_ref_idx = {ch: i for i, ch in enumerate(non_ref)}

    rows = []
    rhs = []
    for pair_name, tau in pair_taus.items():
        parts = pair_name.split("_")
        if len(parts) != 2:
            continue
        ch_a, ch_b = parts
        if ch_a not in channel_order or ch_b not in channel_order:
            continue
        # Equation: t_a - t_b = tau
        row = np.zeros(len(non_ref))
        if ch_a != ref_ch:
            row[non_ref_idx[ch_a]] = 1.0
        if ch_b != ref_ch:
            row[non_ref_idx[ch_b]] = -1.0
        rows.append(row)
        rhs.append(tau)

    A_mat = np.array(rows)
    b_vec = np.array(rhs)
    delays_non_ref, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

    # Build full delay vector
    ch_idx = {ch: i for i, ch in enumerate(channel_order)}
    mic_delays = np.zeros(n_ch)
    for ch, delay in zip(non_ref, delays_non_ref):
        mic_delays[ch_idx[ch]] = delay

    # Steering vector: d[f, m] = exp(-j 2 pi f_hz t_m)
    freq_hz = np.arange(n_freq) * sr / n_fft
    steer = np.exp(
        -1j * 2 * np.pi * freq_hz[:, np.newaxis] * mic_delays[np.newaxis, :]
    )

    # Phase-normalise to channel 0
    steer = steer * np.exp(-1j * np.angle(steer[:, 0:1]))

    # Unit-normalise per frequency
    steer = steer / (np.linalg.norm(steer, axis=1, keepdims=True) + 1e-12)

    return steer, mic_delays


# ── Null-steering at known interferer directions ──────────────────────

def _add_null_steering_to_noise_cov(
    cov_noise: np.ndarray,
    interferer_azimuths: list,
    calibration: dict,
    n_freq: int,
    n_fft: int,
    sr: int,
    channel_order: list,
    null_strength: float = 1.0,
) -> np.ndarray:
    """
    Augment noise covariance with explicit interferer spatial nulls.

    For each known interferer azimuth, adds the outer product of the
    calibrated steering vector (scaled by local noise power) to the
    noise covariance.  This guarantees the MVDR places spatial nulls
    at those directions.

    R_n' = R_n + sum_i  alpha * P_avg(f) * d_i * d_i^H
    """
    n_ch = cov_noise.shape[1]
    cov_out = cov_noise.copy()

    # Average noise power per frequency band
    avg_power = np.real(np.trace(cov_noise, axis1=1, axis2=2)) / n_ch

    for interf_az in interferer_azimuths:
        try:
            steer, _ = compute_steering_doa_model(
                interf_az, calibration, n_freq, n_fft, sr, channel_order,
            )
        except Exception:
            continue

        scale = null_strength

        # Vectorised outer product: (n_freq, n_ch, n_ch)
        outer = np.einsum('fi,fj->fij', steer, steer.conj())
        cov_out += scale * avg_power[:, None, None] * outer

    return cov_out


# ── MVDR beamforming ──────────────────────────────────────────────────

def mvdr_beamform(
    stft: np.ndarray,
    steer: np.ndarray,
    cov_noise: np.ndarray,
    diagonal_loading: float = 1e-6,
    eps: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply MVDR beamformer.

    w[f] = R_n^{-1} d / (d^H R_n^{-1} d + eps)
    y[f,t] = w[f]^H x[:,f,t]

    Returns (enhanced, weights).
    """
    n_ch, n_freq, n_frames = stft.shape
    enhanced = np.zeros((n_freq, n_frames), dtype=np.complex128)
    weights = np.zeros((n_freq, n_ch), dtype=np.complex128)
    eye = np.eye(n_ch, dtype=np.complex128)

    for f in range(n_freq):
        Rn = cov_noise[f] + diagonal_loading * eye
        d = steer[f]
        try:
            Rn_inv = np.linalg.inv(Rn)
        except np.linalg.LinAlgError:
            Rn_inv = np.linalg.pinv(Rn)

        Rn_inv_d = Rn_inv @ d
        denom = d.conj() @ Rn_inv_d + eps
        w = Rn_inv_d / denom
        weights[f] = w
        enhanced[f, :] = w.conj() @ stft[:, f, :]

    return enhanced, weights


# ── Segment-wise steering helpers ─────────────────────────────────────

def _get_block_azimuths(
    candidate: dict,
    n_frames: int,
    block_size: int = 200,
) -> List[Tuple[int, int, float]]:
    """
    Divide the file into blocks and compute per-block circular-mean
    azimuth from the smoothed DoA track.

    Returns
    -------
    blocks : list of (start_frame, end_frame, azimuth_deg)
    """
    az_track = candidate.get("azimuth_track", [])
    mean_az = float(candidate.get("mean_azimuth", 0.0))

    if not az_track:
        return [(0, n_frames, mean_az)]

    # Build full-length azimuth array (NaN where missing)
    length = min(len(az_track), n_frames)
    az_arr = np.full(n_frames, np.nan)
    for i in range(length):
        val = az_track[i]
        if val is not None:
            az_arr[i] = float(val)

    blocks: List[Tuple[int, int, float]] = []
    for start in range(0, n_frames, block_size):
        end = min(start + block_size, n_frames)
        chunk = az_arr[start:end]
        valid = ~np.isnan(chunk)
        if valid.sum() > 0:
            rad = np.deg2rad(chunk[valid])
            local_az = float(np.degrees(np.arctan2(
                np.mean(np.sin(rad)), np.mean(np.cos(rad))))) % 360.0
        else:
            local_az = mean_az
        blocks.append((start, end, local_az))

    return blocks


# ── Entry point ────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for step 02 (MVDR)."""
    ensure_output_dirs()

    enh_cfg = CFG.get("enhancement", {})
    diag_load = float(enh_cfg.get("mvdr_diagonal_loading", 1e-6))
    steering_mode = enh_cfg.get("steering_mode", "doa_model")
    use_interferer_nulling = enh_cfg.get("use_interferer_nulling", False)
    null_strength = float(enh_cfg.get("null_steering_strength", 1.0))
    segment_steering = enh_cfg.get("segment_steering", False)
    block_frames = int(enh_cfg.get("steering_block_frames", 200))

    stft_params = get_stft_params()
    n_fft = stft_params["n_fft"]
    sr = SAMPLE_RATE
    channel_order = get_channel_order()

    # ── Load tracks ────────────────────────────────────────────────────
    tracks_path = DOA_DIR / "doa_tracks.json"
    if not tracks_path.exists():
        raise FileNotFoundError(f"[member2][mvdr] not found: {tracks_path}")
    with open(tracks_path, "r", encoding="utf-8") as fh:
        scene = json.load(fh)
    candidates = scene.get("candidates", [])

    # ── Optionally include provisional candidates ──────────────────
    if enh_cfg.get("include_provisionals", False):
        min_score = float(enh_cfg.get("provisional_min_score", 0.75))
        min_dur = float(enh_cfg.get("provisional_min_duration_s", 5.0))
        min_sep = float(enh_cfg.get("provisional_min_sep_deg", 0.0))
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
            logger.info("[member2][mvdr] including %d/%d provisionals",
                        len(accepted), len(provs))
            candidates = candidates + accepted

    # ── Honour step-01 dedup manifest if available ──────────────────
    # The manifest contains:
    #   "active"       – candidates to beamform (after dedup)
    #   "null_azimuths" – ALL candidate azimuths (incl. deduped/sidelobe
    #                     provisionals) for null-steering coverage.
    null_azimuth_pool: dict[str, float] = {}  # id → azimuth
    manifest_path = INTERMEDIATE_DIR / "active_candidates.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as _mf:
            manifest = json.load(_mf)
        if isinstance(manifest, dict):
            active_ids = set(manifest.get("active", []))
            null_azimuth_pool = manifest.get("null_azimuths", {})
        else:
            # Legacy: plain list
            active_ids = set(manifest)
        candidates = [c for c in candidates if c.get("id") in active_ids]

    if not candidates:
        logger.warning("[member2][mvdr] no candidates — nothing to do.")
        return
    logger.info("[member2][mvdr] %d candidate(s)  steering=%s",
                len(candidates), steering_mode)

    # ── Load calibration (for doa_model steering) ──────────────────────
    calibration = None
    if steering_mode == "doa_model":
        calib_path = CALIB_DIR / "calibration.json"
        if calib_path.exists():
            with open(calib_path, "r", encoding="utf-8") as fh:
                calibration = json.load(fh)
            if "delay_model" not in calibration:
                logger.warning("[member2][mvdr] calibration has no "
                               "delay_model — falling back to eigen")
                calibration = None
        else:
            logger.warning("[member2][mvdr] calibration.json not found "
                           "— falling back to eigen")

    # ── Load STFT ──────────────────────────────────────────────────────
    stft_path = INTERMEDIATE_DIR / "mixture_stft.npy"
    stft = np.load(str(stft_path))
    n_ch, n_freq, n_frames = stft.shape
    logger.info("[member2][mvdr] STFT %s from %s", stft.shape, stft_path.name)

    mix_power = np.mean(np.abs(stft[0]) ** 2)

    # ── Load all masks ─────────────────────────────────────────────────
    all_masks = {}
    for cand in candidates:
        cid = cand.get("id", "spk00")
        mask_path = INTERMEDIATE_DIR / f"{cid}_mask.npy"
        if mask_path.exists():
            all_masks[cid] = np.load(str(mask_path)).astype(np.float64)
        else:
            logger.warning("[member2][mvdr] mask %s missing — all-ones", cid)
            all_masks[cid] = np.ones((n_freq, n_frames), dtype=np.float64)

    if use_interferer_nulling and calibration is not None and len(candidates) > 1:
        logger.info("[member2][mvdr] null-steering enabled "
                    "(strength=%.1f, %d interferer dirs per candidate)",
                    null_strength, len(candidates) - 1)

    # ── Per-candidate MVDR ─────────────────────────────────────────────
    for cand in candidates:
        cid = cand.get("id", "spk00")
        mean_az = float(cand.get("mean_azimuth", 0.0))
        mask = all_masks[cid]

        # Target covariance
        cov_target = estimate_covariance(stft, mask)

        # Noise covariance
        if use_interferer_nulling and len(candidates) > 1:
            other = [all_masks[c.get("id")] for c in candidates
                     if c.get("id") != cid]
            noise_mask = np.clip(np.maximum.reduce(other), 0.0, 1.0)
        else:
            noise_mask = 1.0 - mask
        cov_noise = estimate_covariance(stft, noise_mask)

        # ── Explicit spatial null-steering ─────────────────────────────
        if use_interferer_nulling and calibration is not None and len(candidates) > 1:
            interf_azs = [float(c.get("mean_azimuth", 0))
                          for c in candidates if c.get("id") != cid]
            cov_noise = _add_null_steering_to_noise_cov(
                cov_noise, interf_azs, calibration,
                n_freq, n_fft, sr, channel_order, null_strength,
            )

        # ── Steering vector ────────────────────────────────────────────
        mic_delays = None
        actual_mode = steering_mode

        if steering_mode == "doa_model" and calibration is not None:
            if segment_steering:
                # Per-block steering: adapt direction across time
                blocks = _get_block_azimuths(cand, n_frames, block_frames)
                enhanced = np.zeros((n_freq, n_frames), dtype=np.complex128)
                weights = None

                for b_start, b_end, b_az in blocks:
                    try:
                        b_steer, mic_delays = compute_steering_doa_model(
                            b_az, calibration, n_freq, n_fft, sr,
                            channel_order,
                        )
                    except Exception:
                        b_steer = compute_steering_eigen(cov_target)

                    b_enh, b_w = mvdr_beamform(
                        stft[:, :, b_start:b_end], b_steer, cov_noise,
                        diagonal_loading=diag_load,
                    )
                    enhanced[:, b_start:b_end] = b_enh
                    if weights is None:
                        weights = b_w

                actual_mode = "doa_model_segment"
                n_blocks = len(blocks)
                az_range = max(b[2] for b in blocks) - min(b[2] for b in blocks)

                # Single steering vector for debug output
                try:
                    steer, _ = compute_steering_doa_model(
                        mean_az, calibration, n_freq, n_fft, sr,
                        channel_order,
                    )
                except Exception:
                    steer = compute_steering_eigen(cov_target)
            else:
                try:
                    steer, mic_delays = compute_steering_doa_model(
                        mean_az, calibration, n_freq, n_fft, sr,
                        channel_order,
                    )
                except Exception as exc:
                    logger.warning("[member2][mvdr] %s doa_model failed "
                                   "(%s) — fallback to eigen", cid, exc)
                    steer = compute_steering_eigen(cov_target)
                    actual_mode = "eigen"

                enhanced, weights = mvdr_beamform(
                    stft, steer, cov_noise, diagonal_loading=diag_load,
                )
        else:
            steer = compute_steering_eigen(cov_target)
            actual_mode = "eigen"
            enhanced, weights = mvdr_beamform(
                stft, steer, cov_noise, diagonal_loading=diag_load,
            )

        enh_power = float(np.mean(np.abs(enhanced) ** 2))
        power_ratio = enh_power / (mix_power + 1e-12)

        # Save enhanced STFT
        out_path = SEPARATED_DIR / f"{cid}_enhanced_stft.npy"
        np.save(str(out_path), enhanced)

        # Save debug NPZ (gated)
        save_debug = CFG.get("pipeline", {}).get("save_debug", False)
        if save_debug:
            debug = dict(
                steering_mode=np.array(actual_mode),
                mean_azimuth=np.array(mean_az),
                mask=mask.astype(np.float32),
                steer=steer,
                weights=weights,
                cov_target=cov_target,
                cov_noise=cov_noise,
                mix_power=np.array(mix_power),
                enh_power=np.array(enh_power),
            )
            if mic_delays is not None:
                debug["relative_mic_delays"] = mic_delays
            np.savez_compressed(
                str(SEPARATED_DIR / f"{cid}_debug.npz"), **debug)

        if actual_mode == "doa_model_segment":
            logger.info("[member2][mvdr] %s: steering=%s  az=%.1f  "
                        "blocks=%d  az_range=%.1f°  power_ratio=%.2f",
                        cid, actual_mode, mean_az, n_blocks,
                        az_range, power_ratio)
        else:
            logger.info("[member2][mvdr] %s: steering=%s  az=%.1f  "
                        "power_ratio=%.2f",
                        cid, actual_mode, mean_az, power_ratio)

    logger.info("[member2][mvdr] done — %d candidate(s)", len(candidates))


if __name__ == "__main__":
    main()
