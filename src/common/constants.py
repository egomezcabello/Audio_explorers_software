"""
src.common.constants – Project-wide constants.

All "magic numbers" and fixed labels live here so every member module
references the same values.
"""

# ── Audio constants ─────────────────────────────────────────────────────
SAMPLE_RATE: int = 44_100
"""Sampling rate in Hz – fixed for every recording in this project."""

N_CHANNELS: int = 4
"""Number of microphone channels on the hearing-aid pair."""

CHANNEL_ORDER: list[str] = ["LF", "LR", "RF", "RR"]
"""
Physical channel labels, index-aligned with the columns of loaded WAV data.
  0 → LF  (Left-Front)
  1 → LR  (Left-Rear)
  2 → RF  (Right-Front)
  3 → RR  (Right-Rear)
"""
