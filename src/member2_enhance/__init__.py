"""
src.member2_enhance – Speech enhancement / separation.

Member 2 is responsible for:
  1. (Optionally) applying WPE dereverberation.
  2. Building DoA-guided time-frequency masks from candidate tracks.
  3. Estimating spatial covariance matrices and steering vectors.
  4. Applying MVDR beamforming per candidate.
  5. Post-filtering and exporting enhanced WAVs.

Pipeline scripts are numbered 00–03 and can be run individually or via
``run_all.py``.
"""
