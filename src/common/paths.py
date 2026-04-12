"""
src.common.paths – Centralised path definitions.

Every file path used across the pipeline is defined here so all members
reference the same locations.  Uses pathlib exclusively.
"""

from pathlib import Path

# ── Project root (two levels up from this file: src/common/paths.py) ───
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ── Top-level directories ──────────────────────────────────────────────
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# ── Data files ─────────────────────────────────────────────────────────
EXAMPLE_MIXTURE_WAV: Path = DATA_DIR / "example_mixture.wav"
MIXTURE_WAV: Path = DATA_DIR / "mixture.wav"

# ── Model cache directories ───────────────────────────────────────────
WHISPER_MODEL_DIR: Path = MODELS_DIR / "whisper"

# ── Output sub-directories ────────────────────────────────────────────
CALIB_DIR: Path = OUTPUTS_DIR / "calib"
DOA_DIR: Path = OUTPUTS_DIR / "doa"
INTERMEDIATE_DIR: Path = OUTPUTS_DIR / "intermediate"
SEPARATED_DIR: Path = OUTPUTS_DIR / "separated"
ANALYSIS_DIR: Path = OUTPUTS_DIR / "analysis"
FINAL_DIR: Path = OUTPUTS_DIR / "final"
FIGURES_DIR: Path = OUTPUTS_DIR / "report" / "figures"


def ensure_output_dirs() -> None:
    """Create every output directory if it does not already exist."""
    for d in (
        CALIB_DIR,
        DOA_DIR,
        INTERMEDIATE_DIR,
        SEPARATED_DIR,
        ANALYSIS_DIR,
        FINAL_DIR,
        FIGURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
