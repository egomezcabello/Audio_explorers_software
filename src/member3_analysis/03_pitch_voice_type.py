#!/usr/bin/env python3
"""
03_pitch_voice_type.py – Pitch estimation and voice-type classification.

Estimates the fundamental frequency (F0) contour for each candidate and
classifies voice type (e.g., male / female / child).

TODO:
    - Implement F0 estimation (e.g., pYIN via librosa, or CREPE).
    - Compute pitch statistics (median, std, range).
    - Classify voice type from pitch stats.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.common.constants import SAMPLE_RATE
from src.common.config import CFG
from src.common.logging_utils import setup_logging
from src.common.paths import ANALYSIS_DIR, INTERMEDIATE_DIR, SEPARATED_DIR, ensure_output_dirs

logger = setup_logging(__name__)


@dataclass
class PitchResult:
    """Holds pitch analysis for one candidate."""
    candidate_id: str = ""
    median_f0_hz: float = 0.0
    std_f0_hz: float = 0.0
    min_f0_hz: float = 0.0
    max_f0_hz: float = 0.0
    voice_type: str = "unknown"  # "male", "female", "child", "unknown"


def estimate_pitch(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Estimate the F0 contour of a speech signal.

    Parameters
    ----------
    audio : np.ndarray
        1-D waveform.
    sr : int
        Sample rate.

    Returns
    -------
    f0 : np.ndarray
        Frame-wise F0 in Hz (0 for unvoiced frames).
    """
    import librosa

    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio.astype(np.float32),
            sr=sr,
            fmin=50,
            fmax=600,
            hop_length=512,
            fill_na=0.0,
        )
        if f0 is None:
            n_frames = max(1, len(audio) // 512)
            return np.zeros(n_frames, dtype=np.float64)
        return np.nan_to_num(f0, nan=0.0).astype(np.float64)
    except Exception as exc:
        logger.warning("pyin failed: %s – returning zeros", exc)
        n_frames = max(1, len(audio) // 512)
        return np.zeros(n_frames, dtype=np.float64)


def classify_voice_type(median_f0: float,
                        male_female_hz: float = 175.0,
                        female_child_hz: float = 260.0) -> str:
    """
    Simple heuristic voice-type classification from median F0.

    Returns ``"male"`` / ``"female"`` / ``"child"`` / ``"unknown"``.

    Thresholds are configurable via config.yaml.
    """
    if median_f0 <= 0:
        return "unknown"
    if median_f0 < male_female_hz:
        return "male"
    if median_f0 < female_child_hz:
        return "female"
    return "child"


def main() -> None:
    """Entry point for step 03 (pitch)."""
    ensure_output_dirs()

    # Read configurable thresholds
    analysis_cfg = CFG.get("analysis", {})
    male_female_hz = float(analysis_cfg.get("pitch_male_female_hz", 175.0))
    female_child_hz = float(analysis_cfg.get("pitch_female_child_hz", 260.0))
    mask_floor = float(analysis_cfg.get("pitch_mask_floor", 0.0))
    pitch_percentile = int(analysis_cfg.get("pitch_percentile", 50))

    wav_files = sorted(SEPARATED_DIR.glob("*_enhanced.wav"))
    if not wav_files:
        logger.warning("No enhanced WAVs found – skipping pitch analysis.")
        return

    results = []

    for wav_path in wav_files:
        cid = wav_path.stem.replace("_enhanced", "")
        logger.info("Pitch analysis on %s …", wav_path.name)

        import soundfile as sf
        audio, sr = sf.read(str(wav_path), dtype="float64")

        f0 = estimate_pitch(audio, sr)
        voiced_idx = np.where(f0 > 0)[0]
        voiced = f0[voiced_idx]

        # Load beamforming mask for weighted statistics
        mask_path = INTERMEDIATE_DIR / f"{cid}_mask.npy"
        weights = None
        if mask_path.exists():
            mask = np.load(str(mask_path))       # (n_freq, n_frames_stft)
            mask_mean = mask.mean(axis=0)         # average across frequency
            n_pitch = len(f0)
            n_stft = len(mask_mean)
            if n_pitch > 0 and n_stft > 0:
                weights_all = np.interp(
                    np.linspace(0, 1, n_pitch),
                    np.linspace(0, 1, n_stft),
                    mask_mean,
                )
                # C3: Apply mask floor — ignore frames with low mask weight
                if mask_floor > 0:
                    low_mask = weights_all < mask_floor
                    weights_all[low_mask] = 0.0

                weights = weights_all[voiced_idx]
                logger.debug("  mask-weighted: %d voiced, mask %s",
                             len(voiced), mask.shape)

        pr = PitchResult(candidate_id=cid)
        if len(voiced) > 0:
            if (weights is not None and len(weights) == len(voiced)
                    and weights.sum() > 1e-12):
                # C2: Weighted percentile (P25 by default) for robustness
                order = np.argsort(voiced)
                sorted_f0 = voiced[order]
                sorted_w = weights[order]
                cum_w = np.cumsum(sorted_w)
                target_frac = pitch_percentile / 100.0
                pr.median_f0_hz = float(
                    sorted_f0[np.searchsorted(cum_w, cum_w[-1] * target_frac)])
                # Weighted std
                w_mean = float(np.average(voiced, weights=weights))
                pr.std_f0_hz = float(
                    np.sqrt(np.average((voiced - w_mean) ** 2,
                                       weights=weights)))
            else:
                pr.median_f0_hz = float(np.percentile(voiced, pitch_percentile))
                pr.std_f0_hz = float(np.std(voiced))
            pr.min_f0_hz = float(np.min(voiced))
            pr.max_f0_hz = float(np.max(voiced))
        pr.voice_type = classify_voice_type(
            pr.median_f0_hz, male_female_hz, female_child_hz)

        results.append(pr)
        logger.info("  → median F0=%.1f Hz, type=%s%s",
                     pr.median_f0_hz, pr.voice_type,
                     " (mask-weighted)" if weights is not None else "")

    # Save combined pitch results
    import json
    from dataclasses import asdict
    out_path = ANALYSIS_DIR / "pitch_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2)
    logger.info("Pitch results saved → %s", out_path)
    logger.info("Step 03 (pitch) complete.")


if __name__ == "__main__":
    main()
