"""
src.member1_doa – Calibration, DoA estimation, and tracking.

Member 1 is responsible for:
  1. Loading the 4-channel WAV files and verifying shape / channel order.
  2. Computing and saving STFT representations.
  3. Calibrating microphone array templates (GCC-PHAT).
  4. Estimating direction-of-arrival (DoA) heatmaps.
  5. Tracking and clustering DoA estimates into candidate talkers.

Pipeline scripts are numbered 00–03 and can be run individually or via
``run_all.py``.
"""
