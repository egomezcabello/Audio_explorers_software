"""
test_json_schema.py – Smoke test: verify shared JSON schema helpers.

Run with:
    pytest tests/test_json_schema.py -v
"""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest

from src.common.json_schema import (
    Candidate,
    CandidateOutputs,
    SceneSummary,
    STFTParams,
    load_json,
    save_json,
    validate_scene_summary,
)


def test_scene_summary_defaults() -> None:
    """A default SceneSummary should have correct types."""
    s = SceneSummary()
    assert s.sample_rate == 44_100
    assert s.channel_order == ["LF", "LR", "RF", "RR"]
    assert isinstance(s.stft_params, STFTParams)
    assert isinstance(s.candidates, list)
    assert len(s.candidates) == 0


def test_candidate_defaults() -> None:
    """A default Candidate should have empty track and segments."""
    c = Candidate()
    assert c.id == "spk00"
    assert c.doa_track == []
    assert c.active_segments == []


def test_save_and_load_json(tmp_path: Path) -> None:
    """save_json / load_json round-trip should preserve data."""
    summary = SceneSummary(
        candidates=[
            Candidate(id="spk00", doa_track=[[0, 45.0], [1, 46.0]]),
            Candidate(id="spk01"),
        ]
    )
    path = tmp_path / "test_summary.json"
    save_json(summary, path)

    loaded = load_json(path)
    assert loaded["sample_rate"] == 44_100
    assert len(loaded["candidates"]) == 2
    assert loaded["candidates"][0]["id"] == "spk00"


def test_validate_scene_summary_ok() -> None:
    """A well-formed dict should pass validation."""
    data = asdict(SceneSummary())
    assert validate_scene_summary(data) is True


def test_validate_scene_summary_missing_key() -> None:
    """A dict missing a required key should raise ValueError."""
    data = {"sample_rate": 44_100}  # missing channel_order, etc.
    with pytest.raises(ValueError, match="Missing keys"):
        validate_scene_summary(data)


def test_validate_scene_summary_bad_candidates() -> None:
    """'candidates' must be a list."""
    data = {
        "sample_rate": 44_100,
        "channel_order": ["LF", "LR", "RF", "RR"],
        "stft_params": {},
        "candidates": "not-a-list",
    }
    with pytest.raises(ValueError, match="must be a list"):
        validate_scene_summary(data)
