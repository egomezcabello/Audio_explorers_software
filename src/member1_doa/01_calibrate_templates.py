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
    (0°, 90°, 180°, 270°) using a least-squares cost match against
    geometry-predicted delays.
4.  **Fit a sinusoidal delay model** per pair from the calibration
    templates.  Each pair's delay is modelled as
    τ(θ) = a·cos(θ) + b·sin(θ), which is physically motivated by the
    plane-wave far-field model τ = d⃗·n̂ / c.
5.  Save **templates** (learned median TDOAs), **geometry_templates**
    (expected from mic positions), and the fitted sinusoidal model
    coefficients so step 02 can use them to build corrected steering
    vectors.

GCC-PHAT implementation note
-----------------------------
Each STFT frame already represents the DFT of a windowed time segment.
To estimate the TDOA for a single frame we:
  a. Compute the cross-spectrum G = X₁·X₂*
  b. Apply PHAT whitening: G / |G|
  c. Apply a frequency-band weight (soft bandpass) to emphasise the
     speech band while keeping all DFT bins so the inverse transform
     relationship is preserved.
  d. Use ``np.fft.irfft`` with zero-padding (8×) to over-sample the
     cross-correlation and improve sub-sample delay resolution.
  e. Find the peak within ±max_lag and refine with parabolic
     interpolation.

This avoids the previous bug where frequency-masked bins were stuffed
into a short array and fed to ``irfft``, destroying the DFT alignment
and producing delay estimates ~100× too small.

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
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

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


# ── GCC-PHAT (frame-level, corrected) ─────────────────────────────────

def _gcc_phat_frame(
    X1: np.ndarray,
    X2: np.ndarray,
    freq_bins_hz: np.ndarray,
    freq_range: Tuple[float, float] = (300.0, 8000.0),
    max_tau_sec: float = 0.001,
    zeropad_factor: int = 8,
) -> Tuple[float, float]:
    """
    Estimate TDOA for a single STFT frame using GCC-PHAT.

    The inputs X1, X2 must be the **full** STFT spectrum (all n_fft/2+1
    non-negative frequency bins), not a frequency-masked subset.  A soft
    bandpass weight is applied internally so the DFT bin alignment is
    preserved for the inverse transform.

    Parameters
    ----------
    X1, X2 : np.ndarray, shape (n_freq,)
        Single-frame STFT spectra of two channels.
        n_freq = n_fft // 2 + 1  (the non-negative half).
    freq_bins_hz : np.ndarray, shape (n_freq,)
        Frequency in Hz for each bin.
    freq_range : tuple of float
        (low, high) Hz — bandpass range for the GCC-PHAT.
    max_tau_sec : float
        Maximum plausible delay in seconds.
    zeropad_factor : int
        Zero-padding factor for sub-sample resolution.  E.g. 8 means
        the cross-correlation is computed at 8x the original sample
        rate, giving a delay resolution of 1/(8*sr).

    Returns
    -------
    tdoa_sec : float
        Estimated TDOA in seconds (positive => sound arrives at mic 1
        first, i.e. the wavefront reaches mic 1 before mic 2).
    confidence : float
        Peak height of the normalised GCC-PHAT (0-1 range).
    """
    # Cross-spectrum
    G = X1 * np.conj(X2)
    mag = np.abs(G) + 1e-12
    G_phat = G / mag

    # Soft bandpass weight: raised-cosine edges so we don't create
    # spectral artefacts from a hard cutoff.
    f_lo, f_hi = freq_range
    tw = 100.0  # transition width in Hz
    bp_weight = np.ones(len(freq_bins_hz), dtype=np.float64)
    for i, f in enumerate(freq_bins_hz):
        if f < f_lo - tw:
            bp_weight[i] = 0.0
        elif f < f_lo:
            bp_weight[i] = 0.5 * (1.0 + np.cos(np.pi * (f_lo - f) / tw))
        elif f <= f_hi:
            bp_weight[i] = 1.0
        elif f < f_hi + tw:
            bp_weight[i] = 0.5 * (1.0 + np.cos(np.pi * (f - f_hi) / tw))
        else:
            bp_weight[i] = 0.0
    G_phat = G_phat * bp_weight

    # Zero-padded IRFFT to get over-sampled cross-correlation.
    # irfft(x, n) expects x to be the first n//2+1 non-negative
    # frequency bins.  Our G_phat has length n_freq = n_fft//2+1, so
    # the original signal had n_fft samples.  We set n = n_fft * zeropad
    # to get zeropad x oversampling.
    n_fft_orig = 2 * (len(G_phat) - 1)   # recover original n_fft
    n_out = n_fft_orig * zeropad_factor
    cc = np.fft.irfft(G_phat, n=n_out)
    # cc[k] corresponds to lag k samples at the oversampled rate.
    # Positive lags at the beginning, negative lags wrapped to end.
    # Effective sample rate = sr * zeropad_factor.

    sr_eff = SAMPLE_RATE * zeropad_factor
    max_lag = int(np.ceil(max_tau_sec * sr_eff))
    max_lag = min(max_lag, n_out // 2 - 1)

    # Extract the region of interest: lags from -max_lag to +max_lag
    lags_neg = cc[-max_lag:]        # lags -max_lag ... -1
    lags_pos = cc[: max_lag + 1]    # lags 0 ... +max_lag
    lags = np.concatenate([lags_neg, lags_pos])
    lag_indices = np.arange(-max_lag, max_lag + 1)

    peak_idx = int(np.argmax(np.abs(lags)))
    peak_val = float(np.abs(lags[peak_idx]))
    coarse_lag = int(lag_indices[peak_idx])

    # Parabolic interpolation for sub-bin accuracy
    if 0 < peak_idx < len(lags) - 1:
        alpha = float(np.abs(lags[peak_idx - 1]))
        beta  = float(np.abs(lags[peak_idx]))
        gamma = float(np.abs(lags[peak_idx + 1]))
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > 1e-12:
            p = 0.5 * (alpha - gamma) / denom
        else:
            p = 0.0
        refined_lag = coarse_lag + p
    else:
        refined_lag = float(coarse_lag)

    tdoa_sec = refined_lag / sr_eff

    # ── Confidence: peak-to-noise ratio ──────────────────────────────
    # Measure how sharp the GCC-PHAT peak is relative to the noise floor
    # in the cross-correlation.  This is invariant to zero-padding and
    # overall normalisation.
    #   confidence ≈ 0  → diffuse field / no coherent source
    #   confidence ≈ 1  → single dominant coherent source
    abs_lags = np.abs(lags)
    # Exclude the peak region (±5 bins) to estimate the noise floor
    excl_lo = max(0, peak_idx - 5)
    excl_hi = min(len(abs_lags), peak_idx + 6)
    noise_mask = np.ones(len(abs_lags), dtype=bool)
    noise_mask[excl_lo:excl_hi] = False
    if noise_mask.sum() > 0:
        noise_floor = float(np.median(abs_lags[noise_mask]))
    else:
        noise_floor = 0.0
    confidence = (peak_val - noise_floor) / (peak_val + 1e-15)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return tdoa_sec, confidence


def compute_per_frame_tdoa(
    stft: np.ndarray,
    freq_range: Tuple[float, float] = (300.0, 8000.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame TDOA features for all 6 mic pairs.

    Parameters
    ----------
    stft : np.ndarray
        Shape ``(n_channels, n_freq, n_frames)`` --- complex STFT.
        Must contain ALL frequency bins (not a subset).
    freq_range : tuple
        Frequency band in Hz (used as a soft bandpass).

    Returns
    -------
    tdoa_features : np.ndarray
        Shape ``(n_frames, 6)`` --- TDOA in seconds for each pair.
    confidence : np.ndarray
        Shape ``(n_frames, 6)`` --- GCC-PHAT peak confidence.
    """
    n_ch, n_freq, n_frames = stft.shape
    freq_bins_hz = np.linspace(0, SAMPLE_RATE / 2, n_freq)

    ch_idx = {name: i for i, name in enumerate(CHANNEL_ORDER)}
    n_pairs = len(ALL_MIC_PAIRS)

    # Physical max delay across all pairs (for the search range)
    max_baseline = 0.0
    for m1, m2 in ALL_MIC_PAIRS:
        d = np.linalg.norm(MIC_POSITIONS[m1] - MIC_POSITIONS[m2])
        max_baseline = max(max_baseline, d)
    max_tau_sec = max_baseline / SPEED_OF_SOUND * 1.3  # 30% margin

    tdoa_features = np.zeros((n_frames, n_pairs), dtype=np.float64)
    confidence    = np.zeros((n_frames, n_pairs), dtype=np.float64)

    for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
        i1, i2 = ch_idx[m1], ch_idx[m2]
        for t in range(n_frames):
            tau, conf = _gcc_phat_frame(
                stft[i1, :, t], stft[i2, :, t],
                freq_bins_hz,
                freq_range=freq_range,
                max_tau_sec=max_tau_sec,
            )
            tdoa_features[t, p_idx] = tau
            confidence[t, p_idx]    = conf

    return tdoa_features, confidence


# ── Geometry helpers ──────────────────────────────────────────────────

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


# ── Clustering and assignment ─────────────────────────────────────────

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute the weighted median of *values* with non-negative *weights*.

    The weighted median is the value *v* that minimises
    sum_i  w_i * |x_i - v|.  Equivalent to the point where the
    cumulative weight crosses 50%.

    Parameters
    ----------
    values : 1-D array
    weights : 1-D array, same length, non-negative

    Returns
    -------
    float
        Weighted median.
    """
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cum_w = np.cumsum(w_sorted)
    half = cum_w[-1] / 2.0
    idx = int(np.searchsorted(cum_w, half))
    idx = min(idx, len(v_sorted) - 1)
    return float(v_sorted[idx])


def cluster_and_assign(
    tdoa_features: np.ndarray,
    confidence: np.ndarray,
    n_clusters: int = 4,
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, Dict[int, float]]:
    """
    Cluster frames by TDOA pattern and assign each cluster to a
    ground-truth azimuth.

    Steps
    -----
    1.  Confidence-weighted K-Means clustering.
    2.  Per-cluster outlier trimming (2 sigma in normalised feature space).
    3.  Confidence-weighted median TDOA templates per cluster.
    4.  Hungarian (optimal) assignment to ground-truth azimuths.

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
    assigned_az : dict
        ``{cluster_index: azimuth_deg, ...}``
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans

    n_frames, n_pairs = tdoa_features.shape

    # ── 1. Normalise features for balanced K-Means ─────────────────────
    mean = tdoa_features.mean(axis=0, keepdims=True)
    std  = tdoa_features.std(axis=0, keepdims=True) + 1e-12
    X_norm = (tdoa_features - mean) / std

    # ── 2. Confidence-weighted K-Means ─────────────────────────────────
    # Mean confidence per frame → used as sample_weight so high-quality
    # frames have more influence on centroid positions.
    frame_conf = confidence.mean(axis=1)  # (n_frames,)

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = km.fit_predict(X_norm, sample_weight=frame_conf)

    logger.debug("── Cluster summary (before trimming) ──")
    for ci in range(n_clusters):
        mask_ci = labels == ci
        n_ci = int(mask_ci.sum())
        c_mean = float(frame_conf[mask_ci].mean()) if n_ci > 0 else 0.0
        logger.debug("  Cluster %d: %d frames, mean_conf=%.3f", ci, n_ci, c_mean)

    # ── 3. Per-cluster outlier trimming + weighted-median templates ─────
    #    For each cluster:
    #      a. Compute distance of each frame to the cluster centroid
    #         (in the normalised feature space).
    #      b. Remove frames beyond 2× the intra-cluster std ("outliers").
    #      c. From the retained frames, compute per-pair templates using
    #         the confidence-weighted median.
    outlier_dist_factor = 2.0
    cluster_medians = np.zeros((n_clusters, n_pairs), dtype=np.float64)

    for ci in range(n_clusters):
        mask_ci = labels == ci
        n_ci = int(mask_ci.sum())

        if n_ci == 0:
            logger.warning("  Cluster %d is empty.", ci)
            continue

        X_ci = X_norm[mask_ci]               # normalised features
        feat_ci = tdoa_features[mask_ci]     # original TDOA values
        conf_ci = frame_conf[mask_ci]        # per-frame mean confidence

        # Distance from centroid (normalised space)
        centroid = km.cluster_centers_[ci]    # (n_pairs,)
        dists = np.linalg.norm(X_ci - centroid, axis=1)

        # Outlier gate: keep frames within mean ± factor*std
        d_mean = float(dists.mean())
        d_std  = float(dists.std()) + 1e-12
        keep = dists <= d_mean + outlier_dist_factor * d_std
        n_kept = int(keep.sum())

        # Ensure we keep at least 50% of cluster or ≥10 frames
        if n_kept < max(10, n_ci // 2):
            keep = np.ones(n_ci, dtype=bool)
            n_kept = n_ci

        feat_kept = feat_ci[keep]
        conf_kept = conf_ci[keep]

        logger.debug("  Cluster %d: trimmed %d -> %d frames  "
                      "(rejected %d outliers, dist_thresh=%.2f)",
                      ci, n_ci, n_kept, n_ci - n_kept,
                      d_mean + outlier_dist_factor * d_std)

        # Confidence-weighted median per pair
        for p_idx in range(n_pairs):
            cluster_medians[ci, p_idx] = _weighted_median(
                feat_kept[:, p_idx], conf_kept,
            )

    # ── 4. Hungarian assignment to ground-truth azimuths ───────────────
    # Cost matrix blends two terms:
    #   a. Geometry distance: squared error between cluster template and
    #      geometry-predicted TDOA (same as before, but now on the
    #      trimmed weighted-median templates).
    #   b. Sign-pattern consistency: for each azimuth, certain pairs
    #      should have positive/negative delays.  We penalise sign
    #      mismatches on the on-ear and lateral pairs.
    geo_delays = _geometry_expected_delays()
    gt_azimuths = EXAMPLE_GT_AZIMUTHS
    n_az = len(gt_azimuths)

    cost = np.zeros((n_clusters, n_az), dtype=np.float64)

    for ci in range(n_clusters):
        for ai, az in enumerate(gt_azimuths):
            geo = geo_delays[str(int(az))]

            # (a) Squared TDOA distance
            sq_dist = 0.0
            for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
                key = f"{m1}_{m2}"
                sq_dist += (cluster_medians[ci, p_idx] - geo[key]) ** 2
            cost[ci, ai] += sq_dist

            # (b) Sign-pattern penalty: if a pair has an expected sign
            #     and the cluster template disagrees, add a penalty.
            #     Only applied to pairs where |geo_delay| > 5µs (i.e.
            #     the expected sign is meaningful, not near-zero).
            sign_penalty = 0.0
            for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
                key = f"{m1}_{m2}"
                g = geo[key]
                c = cluster_medians[ci, p_idx]
                if abs(g) > 5e-6 and np.sign(c) != np.sign(g):
                    # penalty proportional to the magnitude of the
                    # expected delay — bigger baselines matter more
                    sign_penalty += abs(g)
            # Scale sign penalty so it has a comparable magnitude
            # to the squared-distance term
            cost[ci, ai] += sign_penalty * 1e3

    # Optimal assignment (minimise total cost)
    row_idx, col_idx = linear_sum_assignment(cost)

    assigned_az: Dict[int, float] = {}
    for ci, ai in zip(row_idx, col_idx):
        assigned_az[ci] = gt_azimuths[ai]

    # ── 5. Build output templates ──────────────────────────────────────
    templates: Dict[str, Dict[str, float]] = {}
    for ci, az_deg in sorted(assigned_az.items(), key=lambda x: x[1]):
        pair_delays: Dict[str, float] = {}
        for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
            key = f"{m1}_{m2}"
            pair_delays[key] = round(float(cluster_medians[ci, p_idx]), 9)
        templates[str(int(az_deg))] = pair_delays

        mask_ci = labels == ci
        n_in_cluster = int(mask_ci.sum())
        mean_c = float(frame_conf[mask_ci].mean()) if n_in_cluster > 0 else 0.0
        logger.debug(
            "  Cluster %d -> %3.0f deg  (%d frames, mean_conf=%.3f)  "
            "cost=%.2e",
            ci, az_deg, n_in_cluster, mean_c,
            cost[ci, gt_azimuths.index(az_deg)],
        )

    return templates, labels, assigned_az


# ── Sinusoidal delay-model fitting ────────────────────────────────────

def fit_sinusoidal_model(
    templates: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Fit a per-pair sinusoidal delay model from the calibration templates.

    For each mic pair, the physical far-field TDOA satisfies:
        tau(theta) = a * cos(theta) + b * sin(theta)

    where a = d_x / c and b = d_y / c for an effective baseline vector.
    This function fits (a, b) from the 4 calibration angles using
    ordinary least squares, giving a smooth delay function for any theta.

    Parameters
    ----------
    templates : dict
        ``{"0": {"LF_LR": tau, ...}, "90": {...}, ...}``

    Returns
    -------
    model : dict
        ``{"LF_LR": {"a": float, "b": float}, ...}``
        tau(theta) = a * cos(theta_rad) + b * sin(theta_rad)
    """
    angles_deg = sorted([float(k) for k in templates.keys()])
    angles_rad = np.deg2rad(angles_deg)
    # Design matrix: [cos(theta), sin(theta)]
    A = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])

    pair_keys = list(next(iter(templates.values())).keys())
    model: Dict[str, Dict[str, float]] = {}

    for key in pair_keys:
        taus = np.array([templates[str(int(a))][key] for a in angles_deg])
        # Least-squares fit: tau = A @ [a, b]
        coeffs, _, _, _ = np.linalg.lstsq(A, taus, rcond=None)
        model[key] = {
            "a": round(float(coeffs[0]), 12),
            "b": round(float(coeffs[1]), 12),
        }

    return model


# ── Balanced-direction refinement ─────────────────────────────────────

def _refine_balanced_selection(
    tdoa_features: np.ndarray,
    confidence: np.ndarray,
    labels: np.ndarray,
    assigned_az: Dict[int, float],
    target_per_dir: int = 150,
    min_per_dir: int = 40,
    w_conf: float = 0.6,
    w_dist: float = 0.4,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, int], Dict[str, float]]:
    """
    Balanced-frame refinement: keep at most *target_per_dir* best frames
    per direction, ranked by a combined quality score::

        quality = w_conf * norm_confidence  -  w_dist * norm_centroid_distance

    This prevents one direction from dominating the calibration set.

    Parameters
    ----------
    tdoa_features : np.ndarray, shape (n_sel, 6)
        TDOA features for gated frames.
    confidence : np.ndarray, shape (n_sel, 6)
        GCC-PHAT confidence for gated frames.
    labels : np.ndarray, shape (n_sel,)
        Cluster label per frame (from K-Means).
    assigned_az : dict
        ``{cluster_index: azimuth_deg, ...}``
    target_per_dir : int
        Maximum frames to keep per direction.
    min_per_dir : int
        Warn if a direction has fewer frames than this.
    w_conf, w_dist : float
        Weights for the ranking score.

    Returns
    -------
    dir_indices    : dict  {az_str: np.ndarray of selected indices}
    initial_counts : dict  {az_str: int}
    refined_counts : dict  {az_str: int}
    dir_mean_conf  : dict  {az_str: float}
    """
    # Normalise features for distance computation
    mean = tdoa_features.mean(axis=0, keepdims=True)
    std  = tdoa_features.std(axis=0, keepdims=True) + 1e-12
    X_norm = (tdoa_features - mean) / std
    mean_conf = confidence.mean(axis=1)   # per-frame mean confidence

    # Per-cluster centroid in normalised space
    centroids: Dict[int, np.ndarray] = {}
    for ci in assigned_az:
        mask = labels == ci
        centroids[ci] = (X_norm[mask].mean(axis=0) if mask.sum() > 0
                         else np.zeros(tdoa_features.shape[1]))

    dir_indices:    Dict[str, np.ndarray] = {}
    initial_counts: Dict[str, int] = {}
    refined_counts: Dict[str, int] = {}
    dir_mean_conf:  Dict[str, float] = {}

    for ci, az in sorted(assigned_az.items(), key=lambda x: x[1]):
        az_key = str(int(az))
        in_cluster = np.where(labels == ci)[0]
        initial_counts[az_key] = len(in_cluster)

        if len(in_cluster) == 0:
            dir_indices[az_key] = np.array([], dtype=int)
            refined_counts[az_key] = 0
            dir_mean_conf[az_key] = 0.0
            logger.warning("[calib] %s°: 0 frames available", az_key)
            continue

        # Per-frame quality score
        confs = mean_conf[in_cluster]
        dists = np.linalg.norm(X_norm[in_cluster] - centroids[ci], axis=1)

        # Normalise each component to [0, 1]
        c_range = confs.max() - confs.min() + 1e-12
        conf_norm = (confs - confs.min()) / c_range
        d_range = dists.max() - dists.min() + 1e-12
        dist_norm = (dists - dists.min()) / d_range

        # Combined score: higher confidence is better, closer to centroid
        # is better.  Simple weighted sum.
        score = w_conf * conf_norm - w_dist * dist_norm

        n_keep = min(target_per_dir, len(in_cluster))
        top_order = np.argsort(score)[::-1][:n_keep]
        selected = in_cluster[top_order]

        dir_indices[az_key] = selected
        refined_counts[az_key] = len(selected)
        dir_mean_conf[az_key] = float(mean_conf[selected].mean())

        if len(in_cluster) < min_per_dir:
            logger.warning("[calib] %s°: only %d frames (min_recommended=%d)",
                           az_key, len(in_cluster), min_per_dir)

    return dir_indices, initial_counts, refined_counts, dir_mean_conf


# ── Main calibration bundle ──────────────────────────────────────────

def build_calibration_bundle(
    stft: np.ndarray,
    freq_range: Tuple[float, float] = (300.0, 8000.0),
    n_clusters: int = 4,
    tag: str = "example",
) -> Dict[str, Any]:
    """
    Full calibration pipeline: per-frame TDOA → cluster → assign →
    balanced refinement → fit sinusoidal model → save.

    Returns the calibration dict ready for JSON serialisation.
    """
    logger.info("[%s][calib] Computing per-frame GCC-PHAT ...", tag)
    tdoa_features, confidence = compute_per_frame_tdoa(stft, freq_range)
    n_frames_total = tdoa_features.shape[0]

    # Per-pair TDOA stats (DEBUG level — use -v to see)
    for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
        vals = tdoa_features[:, p_idx]
        logger.debug("[%s][calib] %s_%s: med=%.1fus std=%.1fus",
                     tag, m1, m2,
                     np.median(vals) * 1e6, np.std(vals) * 1e6)

    # ── Frame selection: energy gate + confidence quantile gate ─────────
    # Two-stage gating:
    #   1. Energy gate:  reject the quietest frames (silence / noise).
    #   2. Confidence quantile gate:  among the energy-passing frames,
    #      keep only those whose mean GCC-PHAT confidence is in the top
    #      portion.  This is controlled by `calibration_confidence_quantile`
    #      (e.g. 0.50 = keep the top 50% by confidence).
    # Optionally, a hard cap `calibration_max_frames` limits the total
    # number of frames sent to clustering.
    doa_cfg = CFG.get("doa", {})
    energy_quantile = float(doa_cfg.get("calibration_energy_quantile", 0.30))
    conf_quantile   = float(doa_cfg.get("calibration_confidence_quantile", 0.50))
    max_frames_cfg  = doa_cfg.get("calibration_max_frames", None)

    # --- Stage 1: energy gate ---
    frame_energy = np.mean(np.abs(stft) ** 2, axis=(0, 1))  # (n_frames,)
    energy_thresh = float(np.quantile(frame_energy, energy_quantile))
    energy_mask = frame_energy >= energy_thresh
    n_after_energy = int(energy_mask.sum())

    # --- Stage 2: confidence quantile gate (applied *within* energy-passing frames) ---
    mean_conf = confidence.mean(axis=1)  # (n_frames,)
    # Compute the confidence threshold as a quantile over energy-passing
    # frames only, so the threshold adapts to the actual data.
    conf_of_passing = mean_conf[energy_mask]
    conf_thresh = float(np.quantile(conf_of_passing, conf_quantile))
    conf_mask = mean_conf >= conf_thresh

    # Combined mask: must pass BOTH gates
    selection_mask = energy_mask & conf_mask
    n_selected = int(selection_mask.sum())

    logger.info("[%s][calib] gated=%d/%d (energy_q=%.2f, conf_q=%.2f, "
                "conf_thresh=%.3f)",
                tag, n_selected, n_frames_total,
                energy_quantile, conf_quantile, conf_thresh)
    logger.debug("[%s][calib] conf stats: all_mean=%.3f sel_mean=%.3f",
                 tag, float(mean_conf.mean()),
                 float(mean_conf[selection_mask].mean())
                 if n_selected > 0 else 0.0)

    # --- Safety: fall back if selection is too aggressive ---
    min_needed = max(4 * n_clusters, 50)
    if n_selected < min_needed:
        logger.warning("[%s][calib] Only %d frames pass gating (need >= %d), "
                       "falling back to energy-only.", tag, n_selected, min_needed)
        selection_mask = energy_mask
        n_selected = int(selection_mask.sum())
    if n_selected < min_needed:
        logger.warning("[%s][calib] Still only %d, using ALL frames.",
                       tag, n_selected)
        selection_mask = np.ones(n_frames_total, dtype=bool)
        n_selected = n_frames_total

    # --- Optional hard cap on total frames ---
    if max_frames_cfg is not None:
        max_frames = int(max_frames_cfg)
        if n_selected > max_frames:
            sel_indices = np.where(selection_mask)[0]
            sel_confs = mean_conf[sel_indices]
            keep_order = np.argsort(sel_confs)[::-1][:max_frames]
            new_mask = np.zeros(n_frames_total, dtype=bool)
            new_mask[sel_indices[keep_order]] = True
            selection_mask = new_mask
            n_selected = max_frames

    tdoa_sel = tdoa_features[selection_mask]
    conf_sel = confidence[selection_mask]

    # ── Clustering and assignment ──────────────────────────────────────
    templates, labels, assigned_az = cluster_and_assign(
        tdoa_sel, conf_sel, n_clusters)

    # Initial counts per direction
    init_counts: Dict[str, int] = {}
    for ci, az in sorted(assigned_az.items(), key=lambda x: x[1]):
        init_counts[str(int(az))] = int((labels == ci).sum())

    # ── Balanced refinement (optional) ─────────────────────────────────
    # After initial clustering and direction assignment, refine by
    # capping each direction at target_per_dir best frames so that no
    # single direction dominates the calibration set.
    use_refinement = bool(doa_cfg.get("calibration_balanced_refinement", True))
    target_per_dir = int(doa_cfg.get(
        "calibration_target_frames_per_direction", 150))
    min_per_dir = int(doa_cfg.get(
        "calibration_min_frames_per_direction", 40))

    if use_refinement:
        dir_indices, _, refined_counts, dir_mean_conf = \
            _refine_balanced_selection(
                tdoa_sel, conf_sel, labels, assigned_az,
                target_per_dir=target_per_dir,
                min_per_dir=min_per_dir,
            )

        # Recompute templates from the balanced subsets using
        # confidence-weighted median (same method as initial templates,
        # but now on a balanced frame set).
        frame_conf_mean = conf_sel.mean(axis=1)
        refined_templates: Dict[str, Dict[str, float]] = {}
        for az_key, sel_idx in dir_indices.items():
            pair_delays: Dict[str, float] = {}
            if len(sel_idx) > 0:
                for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
                    pair_delays[f"{m1}_{m2}"] = round(float(
                        _weighted_median(tdoa_sel[sel_idx, p_idx],
                                         frame_conf_mean[sel_idx])), 9)
            else:
                # Fallback: use initial (unrefined) template
                pair_delays = templates.get(az_key, {})
            refined_templates[az_key] = pair_delays
        templates = refined_templates

        logger.info("[%s][calib] init=%s | refined=%s",
                    tag,
                    {k: v for k, v in sorted(init_counts.items(),
                     key=lambda x: float(x[0]))},
                    {k: v for k, v in sorted(refined_counts.items(),
                     key=lambda x: float(x[0]))})
        logger.info("[%s][calib] mean_conf=%s", tag,
                    {k: f"{v:.2f}" for k, v in sorted(
                        dir_mean_conf.items(),
                        key=lambda x: float(x[0]))})
    else:
        refined_counts = init_counts
        dir_mean_conf = {}
        frame_conf_mean = conf_sel.mean(axis=1)
        for ci, az in assigned_az.items():
            mask = labels == ci
            dir_mean_conf[str(int(az))] = float(
                frame_conf_mean[mask].mean()) if mask.sum() > 0 else 0.0
        logger.info("[%s][calib] counts=%s (no refinement)", tag, init_counts)

    # ── Geometry reference and quality check ───────────────────────────
    geo_templates = _geometry_expected_delays()

    # Compact quality summary: mean |learned - geo| per direction
    delta_summary: Dict[str, float] = {}
    for az_str in sorted(templates.keys(), key=lambda x: float(x)):
        learned = templates[az_str]
        geo = geo_templates.get(az_str, {})
        deltas = [abs(learned[k] - geo[k]) * 1e6
                  for k in learned if k in geo]
        delta_summary[az_str] = (round(float(np.mean(deltas)), 1)
                                 if deltas else 0.0)
    logger.info("[%s][calib] mean|learned-geo| us = %s", tag, delta_summary)

    # ── Fit sinusoidal delay model ─────────────────────────────────────
    delay_model = fit_sinusoidal_model(templates)
    geo_model = fit_sinusoidal_model(geo_templates)
    for key in delay_model:
        cal = delay_model[key]
        geo = geo_model[key]
        logger.debug("[%s][calib] model %s: cal=(%.1f,%.1f)us "
                     "geo=(%.1f,%.1f)us",
                     tag, key,
                     cal["a"] * 1e6, cal["b"] * 1e6,
                     geo["a"] * 1e6, geo["b"] * 1e6)

    # ── Build output bundle ────────────────────────────────────────────
    bundle = {
        "sample_rate": SAMPLE_RATE,
        "channel_order": list(CHANNEL_ORDER),
        "mic_pairs": [f"{m1}_{m2}" for m1, m2 in ALL_MIC_PAIRS],
        "pair_groups": {f"{m1}_{m2}": PAIR_GROUP[(m1, m2)]
                        for m1, m2 in ALL_MIC_PAIRS},
        "mic_positions": {k: v.tolist() for k, v in MIC_POSITIONS.items()},
        "speed_of_sound": SPEED_OF_SOUND,
        "calibration_source": "example_mixture.wav",
        "n_frames_total": int(n_frames_total),
        "n_frames_gated": int(n_selected),
        "frame_selection": {
            "energy_quantile": energy_quantile,
            "confidence_quantile": conf_quantile,
            "confidence_threshold": round(conf_thresh, 4),
        },
        "n_clusters": n_clusters,
        "n_frames_per_direction_initial": {
            k: int(v) for k, v in init_counts.items()},
        "n_frames_per_direction_refined": {
            k: int(v) for k, v in refined_counts.items()},
        "refinement": {
            "enabled": use_refinement,
            "target_per_direction": target_per_dir,
            "min_per_direction": min_per_dir,
        },
        "templates": templates,
        "geometry_templates": geo_templates,
        "delay_model": delay_model,
        "geometry_delay_model": geo_model,
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
            f"STFT file not found: {stft_path}  -- run step 00 first."
        )

    stft = np.load(str(stft_path))
    logger.info("[%s] Loaded STFT: %s", tag, stft.shape)

    bundle = build_calibration_bundle(stft, freq_range, n_clusters, tag=tag)

    # Save canonical calibration file
    canonical_path = CALIB_DIR / "calibration.json"
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    with open(canonical_path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)

    logger.info("[%s][calib] saved -> %s  (%d dirs, %d pairs)",
                tag, canonical_path,
                len(bundle["templates"]), len(ALL_MIC_PAIRS))


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--tag", default="example",
                    help="Input tag (default: example = example_mixture.wav)")
    main(_p.parse_args().tag)
