"""
src.common.constants – Project-wide constants.

All "magic numbers" and fixed labels live here so every member module
references the same values.
"""

from typing import List

# ── Audio constants ─────────────────────────────────────────────────────
SAMPLE_RATE: int = 44_100
"""Sampling rate in Hz – fixed for every recording in this project."""

N_CHANNELS: int = 4
"""Number of microphone channels on the hearing-aid pair."""

CHANNEL_ORDER: List[str] = ["LF", "LR", "RF", "RR"]
"""
Physical channel labels, index-aligned with the columns of loaded WAV data.
  0 → LF  (Left-Front)
  1 → LR  (Left-Rear)
  2 → RF  (Right-Front)
  3 → RR  (Right-Rear)
"""

# ── STFT defaults (override via config.yaml) ───────────────────────────
DEFAULT_N_FFT: int = 1024
DEFAULT_HOP_LENGTH: int = 256
DEFAULT_WIN_LENGTH: int = 1024
DEFAULT_WINDOW: str = "hann"

# ── File-naming conventions ────────────────────────────────────────────
CALIBRATION_FILE: str = "calibration.json"
DOA_POSTERIORS_FILE: str = "doa_posteriors.npy"
DOA_TRACKS_FILE: str = "doa_tracks.json"
EXAMPLE_STFT_FILE: str = "example_stft.npy"
MIXTURE_STFT_FILE: str = "mixture_stft.npy"
ENHANCED_WAV_TEMPLATE: str = "spk{:02d}_enhanced.wav"
DEBUG_NPZ_TEMPLATE: str = "spk{:02d}_debug.npz"
ANALYSIS_JSON_TEMPLATE: str = "spk{:02d}_analysis.json"
TRANSCRIPT_TEMPLATE: str = "spk{:02d}_transcript.txt"
TALKER_OF_INTEREST_FILE: str = "talker_of_interest.wav"
FINAL_SUMMARY_FILE: str = "final_scene_summary.json"
