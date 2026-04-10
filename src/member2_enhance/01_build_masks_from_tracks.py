#!/usr/bin/env python3
"""
01_build_masks_from_tracks.py – DoA-guided time-frequency masks.
=================================================================
Second step of the Member 2 (Enhancement) pipeline.

Builds a soft time-frequency mask for each candidate talker:

1.  Gate each candidate using ``active_segments`` from Member 1.
2.  Score the DoA posterior with an angular Gaussian centred at the
    candidate azimuth.

    - **Subband mode** (``use_subband_masks``): compute a per-frequency
      narrowband SRP-PHAT alignment using the calibrated delay model.
      The result is a frequency-dependent mask from the start.
    - **Broadband fallback**: compute a 1-D time score from the DoA
      posterior and tile it to all frequencies (original behaviour).

3.  Optionally multiply by spatial weights from pilot steering vectors.
4.  Apply a mask exponent for sharpening.
5.  Normalise across candidates per (freq, time) bin so they compete.

STFT convention: ``(n_channels, n_freq, n_frames)`` — same as Member 1.

Outputs
-------
- ``outputs/intermediate/spkXX_mask.npy``    per candidate
- ``outputs/intermediate/masks_debug.npz``   combined debug file
"""

from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from src.common.config import CFG, get_channel_order, get_stft_params
from src.common.constants import SAMPLE_RATE
from src.common.logging_utils import setup_logging
from src.common.paths import CALIB_DIR, DOA_DIR, INTERMEDIATE_DIR, ensure_output_dirs

logger = setup_logging(__name__)


# ── Constants ──────────────────────────────────────────────────────────

_MIC_PAIRS = [
    ("LF", "LR"), ("RF", "RR"),   # on-ear
    ("LF", "RF"), ("LR", "RR"),   # lateral
    ("LF", "RR"), ("LR", "RF"),   # diagonal
]


# ── Helpers ────────────────────────────────────────────────────────────

def _angular_gaussian(angles_deg: np.ndarray, centre_deg: float,
                      sigma_deg: float) -> np.ndarray:
    """Angular Gaussian on [0, 360) centred at *centre_deg*."""
    diff = (angles_deg - centre_deg + 180.0) % 360.0 - 180.0
    return np.exp(-0.5 * (diff / sigma_deg) ** 2)


def _get_azimuth_per_frame(candidate: dict, n_frames: int) -> np.ndarray:
    """Per-frame azimuth from azimuth_track or doa_track. NaN where absent."""
    az = np.full(n_frames, np.nan)

    az_track = candidate.get("azimuth_track")
    if az_track is not None and len(az_track) > 0:
        for i, val in enumerate(az_track):
            if i < n_frames and val is not None:
                az[i] = val
        return az

    doa_track = candidate.get("doa_track", [])
    for entry in doa_track:
        f_idx = int(entry[0])
        if 0 <= f_idx < n_frames:
            az[f_idx] = entry[1]
    return az


def _build_active_frame_mask(
    candidate: dict,
    n_frames: int,
    hop_length: int,
    sr: int,
) -> np.ndarray:
    """
    Convert active_segments [[start_s, end_s], ...] to a boolean
    per-frame mask.  Falls back to non-NaN azimuth if no segments.
    """
    active = np.zeros(n_frames, dtype=bool)
    segments = candidate.get("active_segments", [])

    if not segments:
        # Fallback: frames with valid azimuth are active
        az = _get_azimuth_per_frame(candidate, n_frames)
        return ~np.isnan(az)

    for seg in segments:
        start_s, end_s = seg[0], seg[1]
        f_start = max(0, int(start_s * sr / hop_length))
        f_end = min(n_frames, int(end_s * sr / hop_length) + 1)
        active[f_start:f_end] = True
    return active


def _compute_raw_time_score(
    azimuth_per_frame: np.ndarray,
    active_mask: np.ndarray,
    posteriors: np.ndarray,
    sigma_deg: float,
    time_smooth_sigma: float,
) -> np.ndarray:
    """
    Raw DoA score per frame for one candidate.

    NOT normalised per candidate — weak candidates stay weak.
    Inactive frames are forced to 0 (before and after smoothing).
    """
    n_grid, n_frames = posteriors.shape
    angles_grid = np.linspace(0, 360, n_grid, endpoint=False)
    score = np.zeros(n_frames, dtype=np.float64)

    for t in range(n_frames):
        if not active_mask[t]:
            continue
        az = azimuth_per_frame[t]
        if np.isnan(az):
            continue
        gauss = _angular_gaussian(angles_grid, az, sigma_deg)
        posterior_frame = posteriors[:, t]
        weighted = float(np.sum(gauss * posterior_frame))
        norm = float(np.sum(gauss)) + 1e-12
        score[t] = weighted / norm

    # Gentle time smoothing
    if time_smooth_sigma > 0:
        score = gaussian_filter1d(score, sigma=time_smooth_sigma)

    # Re-zero inactive frames after smoothing
    score[~active_mask] = 0.0
    return score


def _compute_spatial_weights(
    stft: np.ndarray,
    time_mask: np.ndarray,
    pilot_threshold: float = 0.7,
    min_pilot_frames: int = 20,
) -> np.ndarray:
    """
    Frequency-dependent spatial weights from pilot steering vectors.
    Returns shape (n_freq, n_frames) in [0, 1].
    """
    n_ch, n_freq, n_frames = stft.shape

    pilot_idx = np.where(time_mask > pilot_threshold)[0]
    if len(pilot_idx) < min_pilot_frames:
        pilot_idx = np.where(time_mask > 0.5 * pilot_threshold)[0]
    if len(pilot_idx) < min_pilot_frames:
        return np.ones((n_freq, n_frames), dtype=np.float32)

    # Per-frequency steering vector from pilot cross-spectral matrix
    steer = np.zeros((n_freq, n_ch), dtype=np.complex128)
    for f in range(n_freq):
        x_pilot = stft[:, f, pilot_idx]
        cov = (x_pilot @ x_pilot.conj().T) / len(pilot_idx)
        eigvals, eigvecs = np.linalg.eigh(cov)
        d = eigvecs[:, -1]
        d = d * np.exp(-1j * np.angle(d[0]))
        d = d / (np.linalg.norm(d) + 1e-12)
        steer[f] = d

    # Score each (f, t) by alignment with steering vector
    bf_output = np.einsum('fc,cfn->fn', np.conj(steer), stft)
    bf_power = np.abs(bf_output) ** 2
    total_power = np.sum(np.abs(stft) ** 2, axis=0) + 1e-12

    spatial_weights = bf_power / total_power
    sw_max = np.maximum(spatial_weights.max(axis=1, keepdims=True), 1e-12)
    spatial_weights = spatial_weights / sw_max

    return spatial_weights.astype(np.float32)


# ── Narrowband (subband) DoA scoring ──────────────────────────────────

def _compute_narrowband_doa_score(
    stft: np.ndarray,
    candidate_az_per_frame: np.ndarray,
    active_mask: np.ndarray,
    calibration: dict,
    channel_order: list,
    n_fft: int,
    sr: int,
    pair_weights: dict,
    smooth_bins: int = 8,
) -> np.ndarray:
    """
    Per-frequency DoA alignment score using calibrated steering.

    For each mic pair, computes the PHAT-whitened cross-spectrum and
    steers it at the candidate's per-frame azimuth using the calibrated
    delay model.  The weighted sum of per-pair alignments gives a
    narrowband SRP score per (freq, time) bin.

    Parameters
    ----------
    stft : (n_ch, n_freq, n_frames)
    candidate_az_per_frame : (n_frames,) — NaN where inactive
    active_mask : (n_frames,) bool
    calibration : dict with ``delay_model``
    channel_order : list of channel names
    n_fft, sr : STFT parameters
    pair_weights : dict ``"M1_M2" -> weight``
    smooth_bins : smooth alignment across this many freq bins

    Returns
    -------
    score : (n_freq, n_frames) float32 in [0, 1]
    """
    n_ch, n_freq, n_frames = stft.shape
    freq_hz = np.arange(n_freq) * sr / n_fft

    ch_idx = {ch: i for i, ch in enumerate(channel_order)}
    delay_model = calibration["delay_model"]

    active_idx = np.where(active_mask)[0]
    if len(active_idx) == 0:
        return np.zeros((n_freq, n_frames), dtype=np.float32)

    # Per-frame azimuths for active frames
    az_active = candidate_az_per_frame[active_idx].copy()
    mean_az = float(np.nanmean(candidate_az_per_frame[active_mask]))
    az_active = np.where(np.isnan(az_active), mean_az, az_active)
    theta_rad = np.deg2rad(az_active)
    cos_th = np.cos(theta_rad)
    sin_th = np.sin(theta_rad)

    # Accumulate alignment score one pair at a time (memory-efficient)
    score = np.zeros((n_freq, len(active_idx)), dtype=np.float64)
    w_total = 0.0

    for m1, m2 in _MIC_PAIRS:
        key = f"{m1}_{m2}"
        w = float(pair_weights.get(key, 1.0))
        if w < 1e-6 or key not in delay_model:
            continue
        a = delay_model[key]["a"]
        b = delay_model[key]["b"]

        # Per-active-frame delay
        tau_t = a * cos_th + b * sin_th  # (n_active,)

        # Cross-spectrum for this pair
        i1, i2 = ch_idx[m1], ch_idx[m2]
        X1 = stft[i1][:, active_idx]  # (n_freq, n_active)
        X2 = stft[i2][:, active_idx]
        G = X1 * np.conj(X2)
        G_phat = G / (np.abs(G) + 1e-12)

        # Steering phase: exp(-j 2π f τ)
        phase = np.exp(-1j * 2 * np.pi * freq_hz[:, None] * tau_t[None, :])

        # Per (freq, time) alignment
        score += w * np.real(phase * np.conj(G_phat))
        w_total += w

    if w_total > 0:
        score /= w_total  # normalise to roughly [-1, 1]

    # Map from [-1, 1] to [0, 1]
    score = np.clip((score + 1.0) / 2.0, 0.0, 1.0)

    # Smooth across neighbouring frequency bins
    if smooth_bins > 1:
        score = uniform_filter1d(score, size=smooth_bins, axis=0,
                                 mode='reflect')

    # Place into full (n_freq, n_frames) array
    result = np.zeros((n_freq, n_frames), dtype=np.float32)
    result[:, active_idx] = score.astype(np.float32)

    return result


# ── Entry point ────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for step 01 (mask building)."""
    ensure_output_dirs()

    # ── Config ─────────────────────────────────────────────────────────
    enh_cfg = CFG.get("enhancement", {})
    doa_cfg = CFG.get("doa", {})
    stft_params = get_stft_params()
    hop = stft_params["hop_length"]
    n_fft = stft_params["n_fft"]
    sr = SAMPLE_RATE

    sigma_deg = float(enh_cfg.get("mask_sigma_deg", 15.0))
    time_sigma = float(enh_cfg.get("mask_time_smooth_sigma", 1.0))
    mask_exponent = float(enh_cfg.get("mask_exponent", 1.0))
    use_spatial = enh_cfg.get("use_spatial_masks", False)
    pilot_thresh = float(enh_cfg.get("spatial_pilot_threshold", 0.7))

    # Subband masking config
    use_subband = enh_cfg.get("use_subband_masks", False)
    subband_smooth = int(enh_cfg.get("subband_smooth_bins", 8))
    pair_weights = doa_cfg.get("pair_weights", {})

    # ── Load data ──────────────────────────────────────────────────────
    tracks_path = DOA_DIR / "doa_tracks.json"
    if not tracks_path.exists():
        raise FileNotFoundError(f"[member2][mask] not found: {tracks_path}")
    with open(tracks_path, "r", encoding="utf-8") as fh:
        scene = json.load(fh)
    candidates = scene.get("candidates", [])
    logger.info("[member2][mask] %d candidate(s)", len(candidates))

    post_path = DOA_DIR / "doa_posteriors.npy"
    if not post_path.exists():
        raise FileNotFoundError(f"[member2][mask] not found: {post_path}")
    posteriors = np.load(str(post_path))
    logger.info("[member2][mask] posteriors %s", posteriors.shape)

    wpe_path = INTERMEDIATE_DIR / "mixture_stft_wpe.npy"
    raw_path = INTERMEDIATE_DIR / "mixture_stft.npy"
    stft_path = wpe_path if wpe_path.exists() else raw_path
    stft = np.load(str(stft_path))
    n_ch, n_freq, n_frames = stft.shape
    logger.info("[member2][mask] STFT %s from %s", stft.shape, stft_path.name)

    # ── Load calibration (for subband masking) ─────────────────────────
    calibration = None
    channel_order = get_channel_order()
    if use_subband:
        calib_path = CALIB_DIR / "calibration.json"
        if calib_path.exists():
            with open(calib_path, "r", encoding="utf-8") as fh:
                calibration = json.load(fh)
            if "delay_model" not in calibration:
                logger.warning("[member2][mask] calibration has no "
                               "delay_model — falling back to broadband")
                calibration = None
                use_subband = False
            else:
                logger.info("[member2][mask] subband masking enabled "
                            "(smooth=%d bins)", subband_smooth)
        else:
            logger.warning("[member2][mask] calibration.json not found "
                           "— falling back to broadband")
            use_subband = False

    # ── Phase 1: per-candidate masks ───────────────────────────────────
    all_raw_masks = []
    all_time_scores = []
    all_active = []
    cids = []

    for cand in candidates:
        cid = cand.get("id", "spk00")
        cids.append(cid)

        active = _build_active_frame_mask(cand, n_frames, hop, sr)
        az = _get_azimuth_per_frame(cand, n_frames)
        az[~active] = np.nan  # force inactive outside segments

        if use_subband and calibration is not None:
            # — Narrowband path: per-frequency SRP alignment —
            mask_2d = _compute_narrowband_doa_score(
                stft, az, active, calibration, channel_order,
                n_fft, sr, pair_weights, smooth_bins=subband_smooth,
            )
            # Store a 1-D summary for logging
            time_score = mask_2d.mean(axis=0)
            all_raw_masks.append(mask_2d)
        else:
            # — Broadband fallback: tile time-only score to all freqs —
            time_score = _compute_raw_time_score(
                az, active, posteriors, sigma_deg, time_sigma,
            )
            all_raw_masks.append(
                np.tile(time_score[np.newaxis, :], (n_freq, 1))
            )

        all_time_scores.append(time_score)
        all_active.append(active)
        n_active = int(active.sum())
        logger.info("[member2][mask] %s: active=%d  mean=%.4f  max=%.4f"
                    "  mode=%s",
                    cid, n_active,
                    float(time_score.mean()), float(time_score.max()),
                    "subband" if use_subband else "broadband")

    # ── Phase 2: spatial refinement (only for broadband path) ──────────
    all_spatial_w = []
    if use_spatial and not use_subband:
        logger.info("[member2][mask] applying spatial weights "
                    "(pilot=%.2f)", pilot_thresh)
        for i, (cid, raw_mask) in enumerate(zip(cids, all_raw_masks)):
            time_1d = raw_mask[0, :]
            sw = _compute_spatial_weights(
                stft, time_1d, pilot_threshold=pilot_thresh,
            )
            all_raw_masks[i] = raw_mask * sw
            all_spatial_w.append(sw)
    else:
        all_spatial_w = [None] * len(cids)

    # ── Phase 3: mask exponent ─────────────────────────────────────────
    if mask_exponent != 1.0:
        logger.debug("[member2][mask] exponent=%.1f", mask_exponent)
        all_raw_masks = [
            np.power(np.clip(m, 0.0, None), mask_exponent)
            for m in all_raw_masks
        ]

    # ── Phase 4: cross-candidate competition ───────────────────────────
    # mask_i = raw_i / (sum_j raw_j + eps)
    # Where total energy is negligible, all masks stay 0.
    stacked = np.stack(all_raw_masks, axis=0)   # (n_cand, n_freq, n_frames)
    total = stacked.sum(axis=0, keepdims=True)  # (1, n_freq, n_frames)
    eps = 1e-10
    silence = (total < eps)
    final = np.where(silence, 0.0, stacked / (total + eps))

    all_masks = [final[i].astype(np.float32) for i in range(len(cids))]

    logger.info("[member2][mask] competition done — means: %s",
                ", ".join(f"{c}={float(m.mean()):.4f}"
                          for c, m in zip(cids, all_masks)))

    # ── Save ───────────────────────────────────────────────────────────
    for cid, mask in zip(cids, all_masks):
        np.save(str(INTERMEDIATE_DIR / f"{cid}_mask.npy"), mask)

    debug_data = dict(
        candidate_ids=np.array(cids, dtype=object),
        final_masks=np.stack(all_masks, axis=0),
        raw_masks=np.stack(all_raw_masks, axis=0).astype(np.float32),
        active_frame_masks=np.stack(all_active, axis=0),
        time_scores=np.stack(
            [s.astype(np.float32) for s in all_time_scores], axis=0),
    )
    if use_spatial and all_spatial_w[0] is not None:
        debug_data["spatial_weights"] = np.stack(
            all_spatial_w, axis=0).astype(np.float32)

    np.savez_compressed(
        str(INTERMEDIATE_DIR / "masks_debug.npz"), **debug_data)

    logger.info("[member2][mask] saved %d mask(s) + masks_debug.npz",
                len(all_masks))


if __name__ == "__main__":
    main()
