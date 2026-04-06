"""
src.common.json_schema – Shared JSON schema definitions and helpers.

Defines the "contract" data structures that flow between pipeline members
so everyone writes/reads the same format.

NOTE: These are *documentation-level* schemas expressed as Python dicts and
dataclasses – not full JSON-Schema validators.  A lightweight ``validate``
helper is provided for smoke-testing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.constants import CHANNEL_ORDER, SAMPLE_RATE


# ── Shared schema description (for documentation / comments) ───────────
SCENE_SCHEMA_DOC: str = """
Shared JSON schema for the inter-member data contract:

{
  "sample_rate": 44100,
  "channel_order": ["LF", "LR", "RF", "RR"],
  "stft_params": {
      "n_fft": 1024,
      "hop_length": 256,
      "win_length": 1024,
      "window": "hann"
  },
  "candidates": [
    {
      "id": "spk00",
      "doa_track": [[frame_idx, azimuth_deg], ...],
      "active_segments": [[start_s, end_s], ...],
      "outputs": {
        "enhanced_wav": "outputs/separated/spk00_enhanced.wav",
        "analysis_json": "outputs/analysis/spk00_analysis.json",
        "transcript_txt": "outputs/analysis/spk00_transcript.txt"
      }
    }
  ]
}
"""


# ── Dataclass representations ──────────────────────────────────────────
@dataclass
class CandidateOutputs:
    """Paths to the artefacts produced for one candidate talker."""
    enhanced_wav: str = ""
    analysis_json: str = ""
    transcript_txt: str = ""


@dataclass
class Candidate:
    """A single talker candidate discovered by DoA tracking."""
    id: str = "spk00"
    doa_track: List[List[float]] = field(default_factory=list)
    active_segments: List[List[float]] = field(default_factory=list)
    outputs: CandidateOutputs = field(default_factory=CandidateOutputs)


@dataclass
class STFTParams:
    """STFT parameters – shared across the pipeline."""
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    window: str = "hann"


@dataclass
class SceneSummary:
    """
    Top-level data structure exchanged between members.
    Serialised to / from JSON.
    """
    sample_rate: int = SAMPLE_RATE
    channel_order: List[str] = field(default_factory=lambda: list(CHANNEL_ORDER))
    stft_params: STFTParams = field(default_factory=STFTParams)
    candidates: List[Candidate] = field(default_factory=list)


# ── Serialisation helpers ──────────────────────────────────────────────
def save_json(data: Any, path: Path) -> None:
    """Write *data* (dataclass or dict) to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = asdict(data) if hasattr(data, "__dataclass_fields__") else data
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file and return a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_scene_summary(data: Dict[str, Any]) -> bool:
    """
    Lightweight smoke-test validator for a scene-summary dict.

    Returns True if the required top-level keys exist and have sane types.
    Raises ValueError with a message otherwise.
    """
    required_keys = {"sample_rate", "channel_order", "stft_params", "candidates"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys in scene summary: {missing}")
    if not isinstance(data["candidates"], list):
        raise ValueError("'candidates' must be a list")
    return True
