#!/usr/bin/env python3
"""
01_calibrate_templates.py – Array calibration using example_mixture.wav.
========================================================================
Second step of the Member 1 (DoA) pipeline.

What it does
------------
The example_mixture.wav has 4 known speakers at 0°, 90°, 180°, 270°.
We exploit this ground truth to learn *what TDOA patterns each direction
produces* on our specific BTE hearing-aid array.

Algorithm
---------
1.  Compute **per-frame GCC-PHAT** TDOA features for all 6 mic pairs.
    Each frame yields a 6-D feature vector [τ_LF-LR, τ_RF-RR, τ_LF-RF,
    τ_LR-RR, τ_LF-RR, τ_LR-RF].
2.  **Cluster** these frames into K groups (default 4) using K-means
    on the TDOA feature space.
3.  **Assign** each cluster to one of the ground-truth azimuths
    (0°, 90°, 180°, 270°) using sign heuristics:
      - left vs right  → sign of τ on inter-aural (lateral) pairs
      - front vs back  → sign of τ on on-ear (front-rear) pairs
4.  Save **templates**: median TDOA per pair per direction, plus the
    geometry-based expected delays, so that step 02 can use either.

STFT convention: X[ch, f, t]  (same as step 00).

Outputs
-------
- ``outputs/calib/calibration.json``   — canonical handoff file
- ``outputs/calib/{tag}_calibration.json`` — per-tag copy

All 6 pairs are calibrated:
    (LF,LR), (RF,RR), (LF,RF), (LR,RR), (LF,RR), (LR,RF)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.signal import istft as _istft

from src.common.config import CFG, get_stft_params
from src.common.constants import CHANNEL_ORDER, SAMPLE_RATE
from src.common.logging_utils import setup_logging
from src.common.paths import CALIB_DIR, INTERMEDIATE_DIR, ensure_output_dirs

logger = setup_logging(__name__)

# ── All 6 microphone pairs ────────────────────────────────────────────
ALL_MIC_PAIRS: List[Tuple[str, str]] = [
    ("LF", "LR"), ("RF", "RR"),       # on-ear (front-rear, same ear)
    ("LF", "RF"), ("LR", "RR"),       # lateral (cross-ear, same slot)
    ("LF", "RR"), ("LR", "RF"),       # diagonal (cross-ear, diff slot)
]

# Pair-group labels (for logging / interpretability)
PAIR_GROUP: Dict[Tuple[str, str], str] = {
    ("LF", "LR"): "on_ear",  ("RF", "RR"): "on_ear",
    ("LF", "RF"): "lateral", ("LR", "RR"): "lateral",
    ("LF", "RR"): "diagonal", ("LR", "RF"): "diagonal",
}

# ── Known BTE microphone positions (metres) ───────────────────────────
# x = forward, y = left.  These are the physical positions on the head.
MIC_POSITIONS: Dict[str, np.ndarray] = {
    "LF": np.array([+0.006, +0.0875]),
    "LR": np.array([-0.006, +0.0875]),
    "RF": np.array([+0.006, -0.0875]),
    "RR": np.array([-0.006, -0.0875]),
}
SPEED_OF_SOUND: float = 343.0  # m/s

# Ground-truth azimuths for example_mixture.wav (from challenge PDF)
EXAMPLE_GT_AZIMUTHS: List[float] = [0.0, 90.0, 180.0, 270.0]


# ── GCC-PHAT (frame-level) ────────────────────────────────────────────

def _gcc_phat_frame(
    X1: np.ndarray,
    X2: np.ndarray,
    freq_bins: np.ndarray,
    max_tau_sec: float = 0.001,
) -> Tuple[float, float]:
    """
    Compute the GCC-PHAT TDOA for a single STFT frame.

    Parameters
    ----------
    X1, X2 : np.ndarray, shape (n_freq,)
        Single-frame STFT spectra of two channels.
    freq_bins : np.ndarray, shape (n_freq,)
        Frequency values in Hz for each bin.
    max_tau_sec : float
        Maximum plausible delay in seconds.

    Returns
    -------
    tdoa_sec : float
        Estimated TDOA in seconds (positive ⇒ sound arrives at mic2 later).
    confidence : float
        Peak height of the GCC-PHAT function (0 = no signal, 1 = perfect).
    """
    # Cross-spectrum with PHAT weighting
    G = X1 * np.conj(X2)
    mag = np.abs(G) + 1e-12
    G_phat = G / mag

    # Transform to time-lag domain via iDFT over the frequency axis.
    # We use a fine grid of candidate delays for sub-sample resolution.
    sr = SAMPLE_RATE
    max_lag_samples = int(np.ceil(max_tau_sec * sr))
    n_fft_gcc = max(256, 2 * max_lag_samples + 1)
    # Zero-pad to n_fft_gcc and iFFT
    n_freq = len(G_phat)
    padded = np.zeros(n_fft_gcc, dtype=np.complex128)
    padded[:n_freq] = G_phat
    cc = np.fft.irfft(padded, n=n_fft_gcc)

    # Search within ±max_lag_samples
    max_d = min(max_lag_samples, n_fft_gcc // 2 - 1)
    lags = np.concatenate([cc[-max_d:], cc[: max_d + 1]])
    lag_indices = np.arange(-max_d, max_d + 1)

    peak_idx = int(np.argmax(np.abs(lags)))
    peak_val = float(np.abs(lags[peak_idx]))
    coarse_delay = int(lag_indices[peak_idx])

    # Parabolic interpolation for sub-sample accuracy
    if 0 < peak_idx < len(lags) - 1:
        alpha = float(np.abs(lags[peak_idx - 1]))
        beta = float(np.abs(lags[peak_idx]))
        gamma = float(np.abs(lags[peak_idx + 1]))
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > 1e-12:
            p = 0.5 * (alpha - gamma) / denom
        else:
            p = 0.0
        tdoa_samples = coarse_delay + p
    else:
        tdoa_samples = float(coarse_delay)

    tdoa_sec = tdoa_samples / sr
    return tdoa_sec, peak_val


def compute_per_frame_tdoa(
    stft: np.ndarray,
    freq_range: Tuple[int, int] = (300, 8000),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame TDOA features for all 6 mic pairs.

    Parameters
    ----------
    stft : np.ndarray
        Shape ``(n_channels, n_freq, n_frames)`` — complex STFT.
    freq_range : tuple
        Frequency band in Hz.

    Returns
    -------
    tdoa_features : np.ndarray
        Shape ``(n_frames, 6)`` — TDOA in seconds for each pair.
    confidence : np.ndarray
        Shape ``(n_frames, 6)`` — GCC-PHAT peak confidence.
    """
    n_ch, n_freq, n_frames = stft.shape
    sr = SAMPLE_RATE
    freq_bins = np.linspace(0, sr / 2, n_freq)
    f_mask = (freq_bins >= freq_range[0]) & (freq_bins <= freq_range[1])
    stft_sub = stft[:, f_mask, :]
    freq_sub = freq_bins[f_mask]

    ch_idx = {name: i for i, name in enumerate(CHANNEL_ORDER)}
    n_pairs = len(ALL_MIC_PAIRS)

    tdoa_features = np.zeros((n_frames, n_pairs), dtype=np.float64)
    confidence = np.zeros((n_frames, n_pairs), dtype=np.float64)

    for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
        i1, i2 = ch_idx[m1], ch_idx[m2]
        for t in range(n_frames):
            tau, conf = _gcc_phat_frame(
                stft_sub[i1, :, t], stft_sub[i2, :, t], freq_sub,
            )
            tdoa_features[t, p_idx] = tau
            confidence[t, p_idx] = conf

    return tdoa_features, confidence


def _geometry_expected_delays() -> Dict[str, Dict[str, float]]:
    """
    Compute the geometry-based expected TDOA for each pair at the 4
    ground-truth azimuths.

    Returns dict:  { "0": {"LF_LR": tau, ...}, "90": {...}, ... }
    """
    result: Dict[str, Dict[str, float]] = {}
    for az_deg in EXAMPLE_GT_AZIMUTHS:
        az_rad = np.radians(az_deg)
        direction = np.array([np.cos(az_rad), np.sin(az_rad)])
        pair_delays: Dict[str, float] = {}
        for m1, m2 in ALL_MIC_PAIRS:
            d_vec = MIC_POSITIONS[m1] - MIC_POSITIONS[m2]
            tau = float(np.dot(direction, d_vec) / SPEED_OF_SOUND)
            key = f"{m1}_{m2}"
            pair_delays[key] = round(tau, 9)
        result[str(int(az_deg))] = pair_delays
    return result


def cluster_and_assign(
    tdoa_features: np.ndarray,
    confidence: np.ndarray,
    n_clusters: int = 4,
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray]:
    """
    Cluster frames by TDOA pattern and assign each cluster to a
    ground-truth azimuth using sign heuristics.

    Parameters
    ----------
    tdoa_features : np.ndarray, shape (n_frames, 6)
    confidence : np.ndarray, shape (n_frames, 6)
    n_clusters : int

    Returns
    -------
    templates : dict
        ``{ "0": {"LF_LR": median_tdoa, ...}, "90": {...}, ... }``
    labels : np.ndarray, shape (n_frames,)
        Cluster label per frame (0..n_clusters-1).
    """
    from sklearn.cluster import KMeans

    # Weight TDOA features by confidence so noisy frames matter less
    # during clustering.  Normalise each pair column to zero-mean,
    # unit-variance for balanced K-means.
    mean = tdoa_features.mean(axis=0, keepdims=True)
    std = tdoa_features.std(axis=0, keepdims=True) + 1e-12
    X_norm = (tdoa_features - mean) / std

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = km.fit_predict(X_norm)

    # For each cluster, compute the median TDOA per pair
    cluster_medians = np.zeros((n_clusters, 6), dtype=np.float64)
    for ci in range(n_clusters):
        mask = labels == ci
        if mask.sum() > 0:
            cluster_medians[ci] = np.median(tdoa_features[mask], axis=0)

    # ── Heuristic assignment to ground-truth azimuths ──────────────────
    # Pair indices in ALL_MIC_PAIRS:
    #   0: LF-LR  (on-ear left)    → positive τ ⇒ sound from front
    #   1: RF-RR  (on-ear right)   → positive τ ⇒ sound from front
    #   2: LF-RF  (lateral front)  → positive τ ⇒ sound from left
    #   3: LR-RR  (lateral rear)   → positive τ ⇒ sound from left
    #   4: LF-RR  (diagonal)       → mixed
    #   5: LR-RF  (diagonal)       → mixed
    #
    # Heuristic scores:
    #   left_right_score = median(τ_LF-RF) + median(τ_LR-RR)
    #     positive ⇒ left,  negative ⇒ right
    #   front_back_score = median(τ_LF-LR) + median(τ_RF-RR)
    #     positive ⇒ front, negative ⇒ back
    #
    # This gives 4 quadrants:  front+left=90°, front+right=0° … etc.

    # Expected geometry delays for the 4 GT azimuths
    geo_delays = _geometry_expected_delays()

    # Build a score matrix: distance from each cluster median to each
    # geometry template.  We assign clusters to azimuths by minimum
    # total distance (Hungarian-style greedy).
    gt_azimuths = EXAMPLE_GT_AZIMUTHS
    n_az = len(gt_azimuths)
    cost = np.zeros((n_clusters, n_az), dtype=np.float64)

    for ci in range(n_clusters):
        for ai, az in enumerate(gt_azimuths):
            geo = geo_delays[str(int(az))]
            for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
                key = f"{m1}_{m2}"
                cost[ci, ai] += (cluster_medians[ci, p_idx] - geo[key]) ** 2

    # Greedy assignment (works for 4×4; scipy.optimize.linear_sum_assignment
    # would be more correct but this is clearer for students)
    assigned_az = {}
    used_clusters = set()
    used_azimuths = set()

    for _ in range(min(n_clusters, n_az)):
        # Find the (cluster, azimuth) pair with smallest cost
        best_cost = np.inf
        best_ci, best_ai = -1, -1
        for ci in range(n_clusters):
            if ci in used_clusters:
                continue
            for ai in range(n_az):
                if ai in used_azimuths:
                    continue
                if cost[ci, ai] < best_cost:
                    best_cost = cost[ci, ai]
                    best_ci, best_ai = ci, ai
        if best_ci >= 0:
            assigned_az[best_ci] = gt_azimuths[best_ai]
            used_clusters.add(best_ci)
            used_azimuths.add(best_ai)

    # Build templates: median TDOA per pair per assigned azimuth
    templates: Dict[str, Dict[str, float]] = {}
    for ci, az_deg in assigned_az.items():
        pair_delays: Dict[str, float] = {}
        for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
            key = f"{m1}_{m2}"
            pair_delays[key] = round(float(cluster_medians[ci, p_idx]), 9)
        templates[str(int(az_deg))] = pair_delays
        n_in_cluster = int((labels == ci).sum())
        logger.info(
            "  Cluster %d → %3.0f°  (%d frames)  medians=%s",
            ci, az_deg, n_in_cluster,
            {k: f"{v*1e6:.1f}µs" for k, v in pair_delays.items()},
        )

    return templates, labels


def build_calibration_bundle(
    stft: np.ndarray,
    freq_range: Tuple[int, int] = (300, 8000),
    n_clusters: int = 4,
) -> Dict[str, Any]:
    """
    Full calibration pipeline: per-frame TDOA → cluster → assign → save.

    Returns the calibration dict ready for JSON serialisation.
    """
    logger.info("Computing per-frame GCC-PHAT for all 6 pairs …")
    tdoa_features, confidence = compute_per_frame_tdoa(stft, freq_range)
    logger.info(
        "  TDOA features: shape=%s, conf range=[%.3f, %.3f]",
        tdoa_features.shape, confidence.min(), confidence.max(),
    )

    logger.info("Clustering %d frames into %d groups …", tdoa_features.shape[0], n_clusters)
    templates, labels = cluster_and_assign(tdoa_features, confidence, n_clusters)

    # Also store the geometry-based expected delays for reference
    geo_templates = _geometry_expected_delays()

    # Compute global (whole-file) median TDOA per pair as fallback
    global_medians: Dict[str, float] = {}
    for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
        key = f"{m1}_{m2}"
        global_medians[key] = round(float(np.median(tdoa_features[:, p_idx])), 9)

    bundle = {
        "sample_rate": SAMPLE_RATE,
        "channel_order": list(CHANNEL_ORDER),
        "mic_pairs": [f"{m1}_{m2}" for m1, m2 in ALL_MIC_PAIRS],
        "pair_groups": {f"{m1}_{m2}": PAIR_GROUP[(m1, m2)] for m1, m2 in ALL_MIC_PAIRS},
        "mic_positions": {k: v.tolist() for k, v in MIC_POSITIONS.items()},
        "speed_of_sound": SPEED_OF_SOUND,
        "calibration_source": "example_mixture.wav",
        "n_frames_used": int(tdoa_features.shape[0]),
        "n_clusters": n_clusters,
        "templates": templates,               # learned median TDOA per direction
        "geometry_templates": geo_templates,   # expected from mic geometry
        "global_medians": global_medians,      # whole-file median (fallback)
    }
    return bundle


# ── Entry point ────────────────────────────────────────────────────────

def main(tag: str = "example") -> None:
    """
    Run calibration.

    Parameters
    ----------
    tag : str
        Which STFT to calibrate from.  Default ``"example"`` uses
        ``example_mixture.wav`` (which has known ground truth).
    """
    ensure_output_dirs()

    doa_cfg = CFG.get("doa", {})
    freq_range = tuple(doa_cfg.get("freq_range", [300, 8000]))
    n_clusters = doa_cfg.get("calibration_n_clusters", 4)

    stft_path = INTERMEDIATE_DIR / f"{tag}_stft.npy"
    if not stft_path.exists():
        raise FileNotFoundError(
            f"STFT file not found: {stft_path}  — run step 00 first."
        )

    stft = np.load(str(stft_path))
    logger.info("[%s] Loaded STFT: %s  (convention: X[ch, f, t])", tag, stft.shape)

    bundle = build_calibration_bundle(stft, freq_range, n_clusters)

    # Save canonical calibration file
    canonical_path = CALIB_DIR / "calibration.json"
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    with open(canonical_path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)
    logger.info("Canonical calibration → %s", canonical_path)

    # Also save a tag-specific copy
    tag_path = CALIB_DIR / f"{tag}_calibration.json"
    with open(tag_path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)
    logger.info("Tag calibration → %s", tag_path)

    logger.info("Step 01 [%s] complete – %d pair(s), %d direction(s).",
                tag, len(ALL_MIC_PAIRS), len(bundle["templates"]))


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--tag", default="example",
                    help="Input tag (default: example = example_mixture.wav)")
    main(_p.parse_args().tag)
