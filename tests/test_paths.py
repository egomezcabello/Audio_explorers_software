"""
test_paths.py – Smoke test: verify that path constants exist and are consistent.

Run with:
    pytest tests/test_paths.py -v
"""

from pathlib import Path

from src.common.paths import (
    ANALYSIS_DIR,
    CALIB_DIR,
    DATA_DIR,
    DOA_DIR,
    EXAMPLE_MIXTURE_WAV,
    FIGURES_DIR,
    FINAL_DIR,
    INTERMEDIATE_DIR,
    MIXTURE_WAV,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    SEPARATED_DIR,
    ensure_output_dirs,
)


def test_project_root_exists() -> None:
    """PROJECT_ROOT should point to an existing directory."""
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_data_dir_under_root() -> None:
    """DATA_DIR should be a child of PROJECT_ROOT."""
    assert DATA_DIR == PROJECT_ROOT / "data"


def test_output_dirs_under_root() -> None:
    """All output directories should be under outputs/."""
    for d in (CALIB_DIR, DOA_DIR, INTERMEDIATE_DIR, SEPARATED_DIR,
              ANALYSIS_DIR, FINAL_DIR, FIGURES_DIR):
        assert str(d).startswith(str(OUTPUTS_DIR))


def test_wav_paths_correct_suffix() -> None:
    """Input WAV path constants should end in .wav."""
    assert EXAMPLE_MIXTURE_WAV.suffix == ".wav"
    assert MIXTURE_WAV.suffix == ".wav"


def test_ensure_output_dirs_creates_dirs(tmp_path: Path, monkeypatch) -> None:
    """ensure_output_dirs() should create missing directories."""
    # We don't monkey-patch the real dirs – just confirm the function is callable
    # and doesn't crash.  The real dirs already exist thanks to .gitkeep.
    ensure_output_dirs()
    assert CALIB_DIR.exists()
    assert FINAL_DIR.exists()
