#!/usr/bin/env python3
"""
02_doa_estimate.py – Pair-weighted SRP-PHAT Direction-of-Arrival estimation.
=============================================================================
Third step of the Member 1 (DoA) pipeline.

What it does
------------
Computes a 360-bin angular power spectrum for every STFT frame using all
6 microphone pairs, each weighted by its ``pair_weights`` entry in
``config.yaml``.

Algorithm
---------
1.  **Pre-compute steering delays**: for each azimuth θ ∈ [0°, 360°) and
    each mic pair, compute the expected plane-wave TDOA from the known
    mic geometry.
2.  **Per-frame SRP-PHAT**: for each frame *t*:
      a. Compute the cross-spectrum Ĝ = X₁·X₂* / |X₁·X₂*| (GCC-PHAT
         whitening).
      b. For each θ, phase-steer Ĝ to τ(θ) and sum over the frequency
         band [300, 8000] Hz.
      c. Multiply by the pair weight and accumulate.
    This gives P(θ, t) — the angular power map.
3.  Optionally apply **calibration correction** from the templates in
    ``calibration.json`` (currently used as a validation check, not an
    override, since the geometry model is already good).

STFT convention: X[ch, f, t]  (n_channels, n_freq, n_frames).

Outputs
-------
- ``outputs/doa/{tag}_doa_posteriors.npy`` — array (n_grid, n_frames)
- ``outputs/doa/doa_posteriors.npy``       — canonical (symlink to mixture)
"""

from __future__ import annotations

import json
import logging
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


# ── Pre-compute steering vectors ──────────────────────────────────────

def compute_steering_delays(n_grid: int = 360) -> np.ndarray:
    """
    Compute the expected TDOA for each (azimuth, pair) combination.

    Parameters
    ----------
    n_grid : int
        Number of azimuth bins (e.g. 360 = 1° resolution).

    Returns
    -------
    delays : np.ndarray, shape (n_grid, n_pairs)
        Expected TDOA in seconds.  delays[θ, p] is the delay for
        pair *p* when the source is at azimuth θ degrees.
    """
    azimuths_rad = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    directions = np.stack([np.cos(azimuths_rad), np.sin(azimuths_rad)], axis=1)
    # directions shape: (n_grid, 2)

    delays = np.zeros((n_grid, len(ALL_MIC_PAIRS)), dtype=np.float64)
    for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
        d_vec = MIC_POSITIONS[m1] - MIC_POSITIONS[m2]  # shape (2,)
        # TDOA = d_vec · direction / c
        delays[:, p_idx] = directions @ d_vec / SPEED_OF_SOUND

    return delays


def get_pair_weights() -> np.ndarray:
    """
    Read pair weights from config.yaml.  Returns an array of length 6
    aligned with ALL_MIC_PAIRS.

    If a pair's group is disabled in ``use_pair_groups``, its weight
    is forced to 0.
    """
    doa_cfg = CFG.get("doa", {})

    # Per-pair weights from config
    weight_dict = doa_cfg.get("pair_weights", {})
    # Which groups are enabled?
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
    n_grid: int = 360,
    freq_range: Tuple[int, int] = (300, 8000),
    batch_size: int = 50,
) -> np.ndarray:
    """
    Compute the pair-weighted SRP-PHAT angular power spectrum.

    Parameters
    ----------
    stft : np.ndarray
        Shape ``(n_channels, n_freq, n_frames)`` — complex STFT.
    n_grid : int
        Number of azimuth bins.
    freq_range : tuple
        Band-pass frequency range in Hz.
    batch_size : int
        Number of frames to process at once (controls memory).

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
    freqs_hz = freq_bins[f_mask]         # (n_sub_freq,)
    stft_sub = stft[:, f_mask, :]        # (n_ch, n_sub_freq, n_frames)

    ch_idx = {name: i for i, name in enumerate(CHANNEL_ORDER)}

    # Pre-compute steering delays: (n_grid, n_pairs)
    delays = compute_steering_delays(n_grid)
    pair_weights = get_pair_weights()

    logger.info("  Pair weights: %s",
                {f"{m1}_{m2}": pair_weights[p]
                 for p, (m1, m2) in enumerate(ALL_MIC_PAIRS)})

    # Pre-compute steering phase matrix for each pair
    # phase[p] shape: (n_grid, n_sub_freq)
    #   exp(-j * 2π * f * τ(θ, pair))
    steer_phases = []
    for p_idx in range(len(ALL_MIC_PAIRS)):
        tau = delays[:, p_idx]              # (n_grid,)
        phase = np.exp(-1j * 2 * np.pi * freqs_hz[None, :] * tau[:, None])
        steer_phases.append(phase)          # (n_grid, n_sub_freq)

    # Allocate output
    P = np.zeros((n_grid, n_frames), dtype=np.float64)

    # Process in batches to limit memory
    n_batches = int(np.ceil(n_frames / batch_size))
    for b in range(n_batches):
        t0 = b * batch_size
        t1 = min(t0 + batch_size, n_frames)
        batch_frames = t1 - t0

        for p_idx, (m1, m2) in enumerate(ALL_MIC_PAIRS):
            w = pair_weights[p_idx]
            if w == 0.0:
                continue

            i1, i2 = ch_idx[m1], ch_idx[m2]

            # Cross-spectrum with PHAT whitening
            # X1: (n_sub_freq, batch_frames)
            X1 = stft_sub[i1, :, t0:t1]
            X2 = stft_sub[i2, :, t0:t1]
            G = X1 * np.conj(X2)
            mag = np.abs(G) + 1e-12
            G_phat = G / mag              # (n_sub_freq, batch_frames)

            # Steer and sum over frequency
            # steer: (n_grid, n_sub_freq) @ G_phat: (n_sub_freq, batch_frames)
            # = (n_grid, batch_frames)
            steered = steer_phases[p_idx] @ G_phat   # complex
            P[:, t0:t1] += w * np.real(steered)

        if (b + 1) % 20 == 0 or b == n_batches - 1:
            logger.info("  SRP-PHAT batch %d/%d", b + 1, n_batches)

    return P


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

    # Load STFT
    stft_path = INTERMEDIATE_DIR / f"{tag}_stft.npy"
    if not stft_path.exists():
        raise FileNotFoundError(
            f"STFT file not found: {stft_path}  — run step 00 first."
        )
    stft = np.load(str(stft_path))
    logger.info("[%s] Loaded STFT: %s  (convention: X[ch, f, t])", tag, stft.shape)

    # Load calibration (informational – logged but not yet used as override)
    calib_path = CALIB_DIR / "calibration.json"
    if calib_path.exists():
        with open(calib_path, "r", encoding="utf-8") as fh:
            calib = json.load(fh)
        logger.info("[%s] Loaded calibration (%d templates, %d frames used)",
                    tag, len(calib.get("templates", {})),
                    calib.get("n_frames_used", 0))
    else:
        logger.warning("[%s] No calibration.json found – using geometry only.", tag)
        calib = None

    # Run SRP-PHAT
    logger.info("[%s] Running SRP-PHAT (n_grid=%d, freq=[%d, %d] Hz) …",
                tag, n_grid, *freq_range)
    P = srp_phat(stft, n_grid=n_grid, freq_range=freq_range)
    logger.info("[%s] Angular power map: shape=%s, range=[%.4f, %.4f]",
                tag, P.shape, P.min(), P.max())

    # Normalise so each frame's maximum is 1.0 (makes peak detection easier)
    frame_max = P.max(axis=0, keepdims=True)
    frame_max = np.where(frame_max > 0, frame_max, 1.0)
    P_norm = P / frame_max

    # Save per-tag output
    tag_path = DOA_DIR / f"{tag}_doa_posteriors.npy"
    np.save(str(tag_path), P_norm)
    logger.info("[%s] Saved → %s", tag, tag_path)

    # Also save as canonical "doa_posteriors.npy" for the mixture tag
    if tag == "mixture":
        canonical_path = DOA_DIR / "doa_posteriors.npy"
        np.save(str(canonical_path), P_norm)
        logger.info("[%s] Saved canonical → %s", tag, canonical_path)

    logger.info("Step 02 [%s] complete.", tag)


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--tag", default="mixture",
                    help="Input tag (default: mixture)")
    main(_p.parse_args().tag)
