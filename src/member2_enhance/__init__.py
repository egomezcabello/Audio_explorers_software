"""
src.member2_enhance – Speech enhancement / separation.

Member 2 is responsible for:
  1. Building DoA-guided time-frequency masks from candidate tracks.
  2. Estimating spatial covariance matrices and steering vectors.
  3. Applying MVDR beamforming per candidate.
  4. Post-filtering and exporting enhanced WAVs.

Pipeline scripts are numbered 01–03 and can be run individually or via
``run_all.py``.
"""
