#!/usr/bin/env python3
"""
02_doa_estimate.py – Calibration-corrected SRP-PHAT DoA estimation.
====================================================================
Third step of the Member 1 (DoA) pipeline.

What it does
------------
Computes a 360-bin angular power spectrum for every STFT frame using all
6 microphone pairs, each weighted by its ``pair_weights`` entry in
``config.yaml``.  Optionally combines SRP-PHAT with a delay-domain
template-consistency score for more robust DoA estimation.

Algorithm
---------
1.  **Load calibration** from ``calibration.json``.  The calibration
    provides a fitted sinusoidal delay model per mic pair:
        tau(theta) = a * cos(theta) + b * sin(theta)
    where (a, b) are learned from the example recording.  These
    coefficients encode the *effective acoustic baseline* for each pair,
    which may differ from the nominal geometry due to head shadowing,
    reflections, or mic position uncertainty.

2.  **Build steering delays**: for each azimuth theta in [0, 360) and
    each mic pair, evaluate the calibrated delay model.  If no
    calibration is available, fall back to geometry-based delays.

3.  **Per-frame SRP-PHAT**: for each frame *t*:
      a. Compute the cross-spectrum G = X1*X2* / |X1*X2*| (GCC-PHAT
         whitening) restricted to the speech band.
      b. For each theta, phase-steer G to tau(theta) and sum over the
         frequency band [300, 8000] Hz.
      c. Multiply by the pair weight and accumulate.
    This gives P_srp(theta, t) -- the frequency-domain angular power.

4.  **(Optional) Template-consistency scoring**: for each frame *t*:
      a. Estimate the per-pair TDOA via GCC-PHAT (delay domain).
      b. For each candidate angle theta, compute the expected delay
         pattern from the calibrated model.
      c. Score = exp(-sum_p w_p * (tau_obs_p - tau_pred_p)^2 / (2*sigma^2))
    This gives P_tmpl(theta, t) -- the delay-domain consistency map.

5.  **Hybrid combination**:
      P(theta, t) = alpha * P_srp + (1-alpha) * P_tmpl
    where alpha = 1 - template_score_weight.  The SRP-PHAT and
    template-consistency scores provide complementary information:
    SRP works in the frequency domain (phase matching), while template
    matching works in the delay domain (peak-delay matching).  The
    delay-domain score penalises cases where SRP is high due to partial
    frequency matches but the actual delay doesn't match.

Calibration use
---------------
The calibration is used *structurally*, not cosmetically:
  - The steering delays come from the calibrated model, not raw geometry.
  - This means the SRP-PHAT peaks should align better with true source
    directions, because the model accounts for acoustic effects that
    pure geometry ignores (head shadowing, diffraction, etc.).
  - If calibration is missing, the code falls back to geometry-only
    delays and logs a warning.

Pair weighting rationale
------------------------
  - **on_ear** (LF-LR, RF-RR): weight 1.0.  12 mm front-back baseline.
    These are the ONLY pairs that can distinguish front from back.
  - **lateral** (LF-RF, LR-RR): weight 0.3.  175 mm left-right baseline.
    Strong left/right cue but creates front/back ambiguity because
    sin(theta) is symmetric about 90/270.  Down-weighted to suppress
    ghost peaks, but NOT dropped -- they still provide useful lateral
    discrimination.
  - **diagonal** (LF-RR, LR-RF): weight 0.8.  ~175 mm baseline at ~86
    degrees from the x-axis.  Mixed evidence: strong left/right plus a
    small front/back component from the 12 mm x-offset.

STFT convention: X[ch, f, t]  (n_channels, n_freq, n_frames).

Outputs
-------
- ``outputs/doa/{tag}_doa_posteriors.npy`` -- array (n_grid, n_frames)
- ``outputs/doa/doa_posteriors.npy``       -- canonical (for mixture)
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.common.config import CFG, get_stft_params
from src.common.constants import CHANNEL_ORDER, SAMPLE_RATE
from src.common.logging_utils import setup_logging
from src.common.paths import CALIB_DIR, DOA_DIR, INTERMEDIATE_DIR, ensure_output_dirs

logger = setup_logging(__name__)

# ── All 6 microphone pairs ────────────────────────────────────────────
ALL_MIC_PAIRS: List[Tuple[str, str]] = [
    ("LF", "LR"), ("RF", "RR"),       # on-ear
    ("LF", "RF"), ("LR", "RR"),       # lateral
    ("LF", "RR"), ("LR", "RF"),       # diagonal
]

# ── Known BTE microphone positions (metres) ───────────────────────────
# x = forward, y = left.
MIC_POSITIONS: Dict[str, np.ndarray] = {
    "LF": np.array([+0.006, +0.0875]),
    "LR": np.array([-0.006, +0.0875]),
    "RF": np.array([+0.006, -0.0875]),
    "RR": np.array([-0.006, -0.0875]),
}
SPEED_OF_SOUND: float = 343.0  # m/s


# ── Steering-delay computation ────────────────────────────────────────

def compute_geometry_delays(n_grid: int = 360) -> np.ndarray:
    """
    Compute geometry-based TDOA for each (azimuth, pair).

    Returns
    -------
    delays : np.ndarray, shape (n_grid, n_pairs)
        Expected TDOA in seconds.
    """
    azimuths_rad = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    directions = np.stack([np.cos(azimuths_rad), np.sin(azimuths_rad)],
                          axis=1)  # (n_grid, 2)

    delays = np.zeros((n_grid, len(ALL_MIC_PAIRS)), dtype=np.float64)
    for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
        d_vec = MIC_POSITIONS[m1] - MIC_POSITIONS[m2]
        delays[:, p_idx] = directions @ d_vec / SPEED_OF_SOUND

    return delays


def compute_calibrated_delays(
    calib: Dict[str, Any],
    n_grid: int = 360,
) -> np.ndarray:
    """
    Compute steering delays from the calibrated sinusoidal delay model.

    The model for each pair is  tau(theta) = a*cos(theta) + b*sin(theta)
    where (a, b) were fitted from the 4 calibration angles.

    Parameters
    ----------
    calib : dict
        Calibration bundle loaded from calibration.json.
    n_grid : int
        Number of azimuth bins.

    Returns
    -------
    delays : np.ndarray, shape (n_grid, n_pairs)
    """
    delay_model = calib.get("delay_model")
    if delay_model is None:
        logger.warning("No delay_model in calibration -- falling back "
                       "to geometry delays.")
        return compute_geometry_delays(n_grid)

    azimuths_rad = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    cos_az = np.cos(azimuths_rad)
    sin_az = np.sin(azimuths_rad)

    delays = np.zeros((n_grid, len(ALL_MIC_PAIRS)), dtype=np.float64)
    for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
        key = f"{m1}_{m2}"
        if key not in delay_model:
            # Fall back to geometry for this pair
            d_vec = MIC_POSITIONS[m1] - MIC_POSITIONS[m2]
            delays[:, p_idx] = (d_vec[0] * cos_az +
                                d_vec[1] * sin_az) / SPEED_OF_SOUND
            logger.warning("  Pair %s missing from delay_model, "
                           "using geometry.", key)
        else:
            a = delay_model[key]["a"]
            b = delay_model[key]["b"]
            delays[:, p_idx] = a * cos_az + b * sin_az

    return delays


def get_pair_weights() -> np.ndarray:
    """
    Read pair weights from config.yaml.  Returns an array of length 6
    aligned with ALL_MIC_PAIRS.

    If a pair's group is disabled in ``use_pair_groups``, its weight
    is forced to 0.
    """
    doa_cfg = CFG.get("doa", {})

    weight_dict = doa_cfg.get("pair_weights", {})
    group_enabled = doa_cfg.get("use_pair_groups", {
        "on_ear": True, "lateral": True, "diagonal": True,
    })

    pair_group_map = {
        ("LF", "LR"): "on_ear",  ("RF", "RR"): "on_ear",
        ("LF", "RF"): "lateral", ("LR", "RR"): "lateral",
        ("LF", "RR"): "diagonal", ("LR", "RF"): "diagonal",
    }

    weights = np.zeros(len(ALL_MIC_PAIRS), dtype=np.float64)
    for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
        key = f"{m1}_{m2}"
        group = pair_group_map[(m1, m2)]
        if not group_enabled.get(group, True):
            weights[p_idx] = 0.0
        else:
            weights[p_idx] = float(weight_dict.get(key, 1.0))

    return weights


# ── SRP-PHAT core ─────────────────────────────────────────────────────

def srp_phat(
    stft: np.ndarray,
    delays: np.ndarray,
    n_grid: int = 360,
    freq_range: Tuple[int, int] = (300, 8000),
    batch_size: int = 50,
) -> np.ndarray:
    """
    Compute the pair-weighted SRP-PHAT angular power spectrum.

    Parameters
    ----------
    stft : np.ndarray
        Shape ``(n_channels, n_freq, n_frames)`` -- complex STFT.
    delays : np.ndarray
        Shape ``(n_grid, n_pairs)`` -- steering delays in seconds.
        These may come from geometry or from calibration.
    n_grid : int
        Number of azimuth bins.
    freq_range : tuple
        Band-pass frequency range in Hz.
    batch_size : int
        Frames per processing batch (memory control).

    Returns
    -------
    P : np.ndarray, shape (n_grid, n_frames)
        Angular power map.  Higher values indicate a likely source.
    """
    n_ch, n_freq, n_frames = stft.shape
    sr = SAMPLE_RATE
    freq_bins = np.linspace(0, sr / 2, n_freq)

    # Frequency mask for the speech band
    f_mask = (freq_bins >= freq_range[0]) & (freq_bins <= freq_range[1])
    freqs_hz = freq_bins[f_mask]          # (n_sub_freq,)
    stft_sub = stft[:, f_mask, :]         # (n_ch, n_sub_freq, n_frames)

    ch_idx = {name: i for i, name in enumerate(CHANNEL_ORDER)}
    pair_weights = get_pair_weights()

    logger.debug("  Pair weights: %s",
                {f"{m1}_{m2}": pair_weights[p]
                 for p, (m1, m2) in enumerate(ALL_MIC_PAIRS)})

    # Pre-compute steering phase matrix for each pair
    # phase[p] shape: (n_grid, n_sub_freq)
    #   exp(-j * 2*pi * f * tau(theta, pair))
    steer_phases = []
    for p_idx in range(len(ALL_MIC_PAIRS)):
        tau = delays[:, p_idx]              # (n_grid,)
        phase = np.exp(-1j * 2 * np.pi * freqs_hz[None, :] * tau[:, None])
        steer_phases.append(phase)

    # Allocate output
    P = np.zeros((n_grid, n_frames), dtype=np.float64)

    # Process in batches to limit memory
    n_batches = int(np.ceil(n_frames / batch_size))
    for b in range(n_batches):
        t0 = b * batch_size
        t1 = min(t0 + batch_size, n_frames)

        for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
            w = pair_weights[p_idx]
            if w == 0.0:
                continue

            i1, i2 = ch_idx[m1], ch_idx[m2]

            # Cross-spectrum with PHAT whitening
            X1 = stft_sub[i1, :, t0:t1]
            X2 = stft_sub[i2, :, t0:t1]
            G = X1 * np.conj(X2)
            mag = np.abs(G) + 1e-12
            G_phat = G / mag

            # Steer and sum over frequency
            steered = steer_phases[p_idx] @ G_phat   # (n_grid, batch)
            P[:, t0:t1] += w * np.real(steered)

        if (b + 1) % 20 == 0 or b == n_batches - 1:
            logger.debug("  SRP-PHAT batch %d/%d", b + 1, n_batches)

    return P


# ── Per-frame GCC-PHAT delay estimation (for template matching) ──────

def _gcc_phat_delay_frame(
    X1: np.ndarray,
    X2: np.ndarray,
    freq_bins_hz: np.ndarray,
    freq_range: Tuple[float, float] = (300.0, 8000.0),
    max_tau_sec: float = 0.001,
    zeropad_factor: int = 4,
) -> Tuple[float, float]:
    """
    Fast GCC-PHAT delay estimate for a single frame.

    Simplified version (no parabolic refinement, lower zero-padding)
    for use in the template-consistency scorer.

    Returns
    -------
    tdoa_sec : float
        Estimated delay in seconds.
    confidence : float
        Peak-to-noise ratio of the cross-correlation (0 to 1).
    """
    G = X1 * np.conj(X2)
    mag = np.abs(G) + 1e-12
    G_phat = G / mag

    # Soft bandpass
    f_lo, f_hi = freq_range
    tw = 100.0
    bp = np.ones(len(freq_bins_hz), dtype=np.float64)
    lo_mask = freq_bins_hz < f_lo
    lo_edge = (freq_bins_hz >= f_lo - tw) & (freq_bins_hz < f_lo)
    hi_edge = (freq_bins_hz > f_hi) & (freq_bins_hz <= f_hi + tw)
    hi_mask = freq_bins_hz > f_hi + tw
    bp[lo_mask & ~lo_edge] = 0.0
    bp[lo_edge] = 0.5 * (1.0 + np.cos(np.pi * (f_lo - freq_bins_hz[lo_edge]) / tw))
    bp[hi_edge] = 0.5 * (1.0 + np.cos(np.pi * (freq_bins_hz[hi_edge] - f_hi) / tw))
    bp[hi_mask] = 0.0
    G_phat *= bp

    n_fft_orig = 2 * (len(G_phat) - 1)
    n_out = n_fft_orig * zeropad_factor
    cc = np.fft.irfft(G_phat, n=n_out)

    sr_eff = SAMPLE_RATE * zeropad_factor
    max_lag = int(np.ceil(max_tau_sec * sr_eff))
    max_lag = min(max_lag, n_out // 2 - 1)

    lags = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])
    lag_indices = np.arange(-max_lag, max_lag + 1)
    abs_lags = np.abs(lags)
    peak_idx = int(np.argmax(abs_lags))
    peak_val = float(abs_lags[peak_idx])

    # Confidence: peak-to-noise ratio (same formula as Step 01)
    excl_lo = max(0, peak_idx - 5)
    excl_hi = min(len(abs_lags), peak_idx + 6)
    noise_mask = np.ones(len(abs_lags), dtype=bool)
    noise_mask[excl_lo:excl_hi] = False
    noise_floor = float(np.median(abs_lags[noise_mask])) if noise_mask.sum() > 0 else 0.0
    conf = float(np.clip((peak_val - noise_floor) / (peak_val + 1e-15), 0.0, 1.0))

    return float(lag_indices[peak_idx]) / sr_eff, conf


def compute_frame_delays(
    stft: np.ndarray,
    freq_range: Tuple[float, float] = (300.0, 8000.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame TDOA and confidence for all 6 pairs.

    Used by the template-consistency scorer so that low-confidence
    pair delays can be down-weighted.

    Parameters
    ----------
    stft : np.ndarray, shape (n_ch, n_freq, n_frames)
    freq_range : tuple

    Returns
    -------
    delays : np.ndarray, shape (n_frames, 6)
        Estimated TDOA per frame per pair.
    confidences : np.ndarray, shape (n_frames, 6)
        GCC-PHAT peak-to-noise confidence per frame per pair (0–1).
    """
    n_ch, n_freq, n_frames = stft.shape
    freq_bins_hz = np.linspace(0, SAMPLE_RATE / 2, n_freq)
    ch_idx = {name: i for i, name in enumerate(CHANNEL_ORDER)}

    max_baseline = max(
        np.linalg.norm(MIC_POSITIONS[m1] - MIC_POSITIONS[m2])
        for m1, m2 in ALL_MIC_PAIRS
    )
    max_tau = max_baseline / SPEED_OF_SOUND * 1.3

    delays = np.zeros((n_frames, len(ALL_MIC_PAIRS)), dtype=np.float64)
    confidences = np.zeros((n_frames, len(ALL_MIC_PAIRS)), dtype=np.float64)
    for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
        i1, i2 = ch_idx[m1], ch_idx[m2]
        for t in range(n_frames):
            tau, conf = _gcc_phat_delay_frame(
                stft[i1, :, t], stft[i2, :, t],
                freq_bins_hz, freq_range=freq_range,
                max_tau_sec=max_tau,
            )
            delays[t, p_idx] = tau
            confidences[t, p_idx] = conf
    return delays, confidences


def template_consistency_map(
    frame_delays: np.ndarray,
    model_delays: np.ndarray,
    pair_weights: np.ndarray,
    sigma_sec: float = 25e-6,
    frame_confidences: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute template-consistency score for each (angle, frame).

    For each frame, the observed per-pair delays are compared against
    the model-predicted delays.  A Gaussian kernel converts delay
    mismatch into a [0, 1] score.

    Why sigma matters
    -----------------
    The sigma controls how "forgiving" the template match is.  With
    real overlapping speech, the per-frame GCC-PHAT delay can shift by
    10–30 µs even for a dominant source, so sigma should be at least
    ~25 µs to avoid under-scoring valid frames.  A sigma of 15 µs was
    previously used but produced a very sparse template map (mean ≈ 0.007)
    that barely contributed to the hybrid score.

    Confidence weighting
    --------------------
    If ``frame_confidences`` is provided (shape: n_frames × n_pairs),
    each pair's mismatch contribution is scaled by its GCC-PHAT
    confidence.  This means unreliable delay estimates (low confidence)
    penalise the score less, making the template map more robust to
    noisy pairs.

    Parameters
    ----------
    frame_delays : np.ndarray, shape (n_frames, n_pairs)
        Observed per-frame TDOA from GCC-PHAT.
    model_delays : np.ndarray, shape (n_grid, n_pairs)
        Predicted TDOA from the calibrated sinusoidal model for each
        azimuth.
    pair_weights : np.ndarray, shape (n_pairs,)
        Relative importance of each pair.
    sigma_sec : float
        Standard deviation of the Gaussian kernel in seconds.
    frame_confidences : np.ndarray or None, shape (n_frames, n_pairs)
        Per-frame per-pair GCC-PHAT confidence (0–1).  If None, all
        pairs are weighted equally (backward-compatible).

    Returns
    -------
    T : np.ndarray, shape (n_grid, n_frames)
        Template-consistency map.  Higher = better match.
    """
    n_grid, n_pairs = model_delays.shape
    n_frames = frame_delays.shape[0]

    # ── Build effective per-frame-per-pair weights ─────────────────────
    # Base weight: the static pair weight (normalised to sum to 1).
    w_base = pair_weights / (pair_weights.sum() + 1e-12)  # (n_pairs,)

    if frame_confidences is not None:
        # Multiply pair weight by the frame-level pair confidence so
        # that unreliable delay estimates contribute less.
        # w_eff shape: (n_frames, n_pairs)
        w_eff = w_base[np.newaxis, :] * frame_confidences
        # Re-normalise per frame so weights sum to 1 across pairs
        w_sum = w_eff.sum(axis=1, keepdims=True) + 1e-12
        w_eff = w_eff / w_sum
        # Reshape for broadcasting: (1, n_frames, n_pairs)
        w_3d = w_eff[np.newaxis, :, :]
    else:
        # Static pair weights only (broadcast over frames)
        w_3d = w_base[np.newaxis, np.newaxis, :]  # (1, 1, n_pairs)

    # ── Compute weighted squared delay mismatch ───────────────────────
    fd = frame_delays[np.newaxis, :, :]   # (1, n_frames, n_pairs)
    md = model_delays[:, np.newaxis, :]   # (n_grid, 1, n_pairs)

    sq_err = (fd - md) ** 2               # (n_grid, n_frames, n_pairs)
    weighted_sq = sq_err * w_3d
    total_sq = weighted_sq.sum(axis=2)     # (n_grid, n_frames)

    # Gaussian kernel
    T = np.exp(-total_sq / (2.0 * sigma_sec ** 2))  # (n_grid, n_frames)

    return T


# ── Peak-finding and comparison helpers ───────────────────────────────

def _angular_dist(a: float, b: float) -> float:
    """Shortest angular distance in degrees (0–180)."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def _find_top_peaks_avg(
    P: np.ndarray, n_peaks: int = 8, min_dist: int = 40,
) -> List[float]:
    """
    Find top-N peaks in the *time-averaged* angular power spectrum.
    Returns sorted list of peak angles in degrees.
    """
    from scipy.signal import find_peaks as _find_peaks
    n_grid = P.shape[0]
    avg = P.mean(axis=1)
    pad = min_dist + 2
    tiled = np.concatenate([avg[-pad:], avg, avg[:pad]])
    peaks, _ = _find_peaks(tiled, distance=min_dist)
    peaks_orig = peaks - pad
    valid = (peaks_orig >= 0) & (peaks_orig < n_grid)
    peaks_orig = peaks_orig[valid]
    heights = tiled[peaks[valid]]
    order = np.argsort(heights)[::-1][:n_peaks]
    return sorted(float(peaks_orig[o]) for o in order)


def _validate_example_doa(
    P: np.ndarray,
    label: str = "SRP-PHAT",
) -> List[float]:
    """
    Compare detected peaks against 0°/90°/180°/270° ground truth.
    Returns per-direction angular errors (compact logging).
    """
    expected = [0.0, 90.0, 180.0, 270.0]
    detected = _find_top_peaks_avg(P, n_peaks=4)

    if not detected:
        logger.info("[example][doa][%s] no peaks found", label)
        return []

    errors: List[float] = []
    for exp in expected:
        best_err = min(_angular_dist(exp, d) for d in detected)
        errors.append(best_err)

    mean_err = sum(errors) / len(errors) if errors else 0.0
    logger.info("[example][doa][%s] peaks=%s | mean_err=%.1f°",
                label, [f"{d:.0f}" for d in detected], mean_err)
    return errors


def _compare_posteriors(
    P_srp: np.ndarray,
    P_norm: np.ndarray,
    tag: str,
) -> None:
    """
    Print a concise descriptive comparison between raw-SRP and hybrid
    posteriors.  No ground truth needed — this is for mixture files.
    """
    srp_peaks = _find_top_peaks_avg(P_srp)
    hyb_peaks = _find_top_peaks_avg(P_norm)

    # Mean angular shift: for each hybrid peak, find nearest SRP peak
    shifts: List[float] = []
    for h in hyb_peaks:
        if srp_peaks:
            shifts.append(min(_angular_dist(h, s) for s in srp_peaks))
    mean_shift = float(np.mean(shifts)) if shifts else 0.0

    # Sharpness: mean per-frame peak value
    hyb_sharpness = float(P_norm.max(axis=0).mean())

    logger.info("[%s][doa] srp_peaks=%s | hybrid_peaks=%s",
                tag,
                [f"{p:.0f}" for p in srp_peaks],
                [f"{p:.0f}" for p in hyb_peaks])
    logger.info("[%s][doa] mean_shift=%.1f° | hybrid_sharpness=%.3f",
                tag, mean_shift, hyb_sharpness)


# ── Entry point ────────────────────────────────────────────────────────

def main(tag: str = "mixture") -> None:
    """
    Run DoA estimation for the given tag.

    Parameters
    ----------
    tag : str
        Input tag, e.g. ``"example"`` or ``"mixture"``.
    """
    ensure_output_dirs()

    doa_cfg = CFG.get("doa", {})
    n_grid = doa_cfg.get("n_grid", 360)
    freq_range = tuple(doa_cfg.get("freq_range", [300, 8000]))

    # Hybrid scoring config
    use_hybrid = bool(doa_cfg.get("use_hybrid_doa_score", True))
    template_weight = float(doa_cfg.get("template_score_weight", 0.30))
    delay_sigma_us = float(doa_cfg.get("delay_mismatch_sigma_us", 25.0))
    delay_sigma_sec = delay_sigma_us * 1e-6

    # Load STFT
    stft_path = INTERMEDIATE_DIR / f"{tag}_stft.npy"
    if not stft_path.exists():
        raise FileNotFoundError(
            f"STFT file not found: {stft_path}  -- run step 00 first."
        )
    stft = np.load(str(stft_path))
    logger.info("[%s] Loaded STFT: %s", tag, stft.shape)

    # Load calibration and build steering delays
    calib_path = CALIB_DIR / "calibration.json"
    if calib_path.exists():
        with open(calib_path, "r", encoding="utf-8") as fh:
            calib = json.load(fh)
        logger.debug("[%s] calibration: delay_model=%s, %d templates, "
                     "%d frames",
                     tag, "delay_model" in calib,
                     len(calib.get("templates", {})),
                     calib.get("n_frames_gated", 0))
        delays = compute_calibrated_delays(calib, n_grid)
    else:
        logger.warning("[%s] No calibration.json — using geometry delays.",
                       tag)
        calib = None
        delays = compute_geometry_delays(n_grid)

    # ── SRP-PHAT ──────────────────────────────────────────────────────
    logger.info("[%s][doa] SRP-PHAT (n_grid=%d, freq=[%d,%d] Hz) ...",
                tag, n_grid, *freq_range)
    P_srp = srp_phat(stft, delays, n_grid=n_grid, freq_range=freq_range)

    # ── Template-consistency scoring (optional) ───────────────────────
    if use_hybrid and calib is not None:
        logger.info("[%s][doa] template-consistency "
                    "(σ=%.0fµs, weight=%.2f) ...",
                    tag, delay_sigma_us, template_weight)
        frame_delays, frame_confs = compute_frame_delays(stft, freq_range)
        pair_weights = get_pair_weights()

        logger.debug("[%s][doa] frame_delay_conf: mean=%.3f min=%.3f "
                     "max=%.3f", tag,
                     float(frame_confs.mean()),
                     float(frame_confs.min()),
                     float(frame_confs.max()))

        P_tmpl = template_consistency_map(
            frame_delays, delays, pair_weights,
            sigma_sec=delay_sigma_sec,
            frame_confidences=frame_confs,
        )
        logger.debug("[%s][doa] template_map: mean=%.4f max=%.4f",
                     tag, P_tmpl.mean(), P_tmpl.max())

        # Normalise SRP to [0, 1] per-frame before combining
        srp_max = P_srp.max(axis=0, keepdims=True)
        srp_max = np.where(srp_max > 0, srp_max, 1.0)
        P_srp_norm = P_srp / srp_max

        # Template scores are already in [0, 1] from Gaussian kernel
        alpha = 1.0 - template_weight
        P_combined = alpha * P_srp_norm + template_weight * P_tmpl

        # Re-normalise to [0, 1] per frame
        comb_max = P_combined.max(axis=0, keepdims=True)
        comb_max = np.where(comb_max > 0, comb_max, 1.0)
        P_norm = P_combined / comb_max
    else:
        if use_hybrid and calib is None:
            logger.warning("[%s] Hybrid requested but no calibration "
                           "— SRP-only fallback.", tag)
        # Normalise per-frame so each frame's maximum is 1.0
        frame_max = P_srp.max(axis=0, keepdims=True)
        frame_max = np.where(frame_max > 0, frame_max, 1.0)
        P_norm = P_srp / frame_max

    # ── Validation / comparison ───────────────────────────────────────
    if tag == "example":
        # Ground truth available — report SRP and hybrid errors
        srp_errors = _validate_example_doa(P_srp, label="srp")
        hybrid_errors = _validate_example_doa(P_norm, label="hybrid")
        if srp_errors and hybrid_errors:
            srp_mean = sum(srp_errors) / len(srp_errors)
            hyb_mean = sum(hybrid_errors) / len(hybrid_errors)
            logger.info("[example][doa] delta(hybrid-srp)=%+.1f°",
                        hyb_mean - srp_mean)
    else:
        # No ground truth — print descriptive comparison only
        _compare_posteriors(P_srp, P_norm, tag)

    # ── Save outputs ──────────────────────────────────────────────────
    # Hybrid (final canonical output)
    tag_path = DOA_DIR / f"{tag}_doa_posteriors.npy"
    np.save(str(tag_path), P_norm)

    # Raw SRP (optional debug file)
    srp_path = DOA_DIR / f"{tag}_doa_posteriors_srp.npy"
    np.save(str(srp_path), P_srp)

    if tag == "mixture":
        canonical_path = DOA_DIR / "doa_posteriors.npy"
        np.save(str(canonical_path), P_norm)

    logger.info("[%s][doa] saved -> %s  (+ srp)", tag, tag_path)


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--tag", default="mixture",
                    help="Input tag (default: mixture)")
    main(_p.parse_args().tag)