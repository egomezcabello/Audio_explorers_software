# Audio Explorers – Hearing-Aid Microphone Array Scene Analysis

> **Status:** scaffold / placeholder — algorithms are **not yet implemented**.

A 4-channel hearing-aid microphone array pipeline that detects, separates,
and analyses multiple talkers in a scene, then selects and enhances the
*talker of interest*.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Setup Instructions](#setup-instructions)
4. [Data Files](#data-files)
5. [Pipeline Overview & Member Responsibilities](#pipeline-overview--member-responsibilities)
6. [Running the Pipeline](#running-the-pipeline)
7. [Output Hand-off Between Members](#output-hand-off-between-members)
8. [Smoke Tests](#smoke-tests)
9. [What Is Implemented vs. Placeholder](#what-is-implemented-vs-placeholder)
10. [Git Branching & Collaboration](#git-branching--collaboration)

---

## Project Overview

The pipeline processes 4-channel hearing-aid recordings through four
sequential stages:

| Stage | Owner | Description |
|---|---|---|
| 1 – DoA | Member 1 | Calibration, Direction-of-Arrival estimation, tracking |
| 2 – Enhancement | Member 2 | WPE, TF masks, MVDR beamforming, post-filter |
| 3 – Analysis | Member 3 | VAD, language ID, Whisper ASR, pitch/voice type |
| 4 – Fusion | Member 4 | Merge candidates, score TOI, figures, final outputs |

**Audio format:** 4 channels, 44 100 Hz, WAV.  
**Channel order (always):** `["LF", "LR", "RF", "RR"]`

---

## Repository Structure

```
Audio_explorers_software/
├── config.yaml                 # Shared pipeline configuration
├── pyproject.toml              # Python project metadata & dependencies
├── requirements-dev.txt        # Dev-only dependencies
├── Makefile                    # Convenience targets
├── .gitignore / .gitattributes
├── .env.example                # Environment variable template
├── LICENSE
├── CONTRIBUTING.md             # PR checklist & collaboration rules
│
├── data/                       # Input audio (NOT committed)
│   ├── example_mixture.wav
│   └── mixture.wav
│
├── models/                     # Pre-trained weights / cache (NOT committed)
│   ├── whisper/
│   ├── speechbrain/
│   ├── optional_separation/
│   └── cache/
│
├── outputs/                    # All generated artefacts (NOT committed)
│   ├── calib/                  #   calibration.json
│   ├── doa/                    #   doa_posteriors.npy, doa_tracks.json
│   ├── intermediate/           #   STFTs, masks
│   ├── separated/              #   spkXX_enhanced.wav, spkXX_debug.npz
│   ├── analysis/               #   spkXX_analysis.json, spkXX_transcript.txt
│   ├── final/                  #   talker_of_interest.wav, final_scene_summary.json
│   └── report/                 #   figures/, tables/, draft/
│
├── src/
│   ├── common/                 # Shared utilities (config, I/O, STFT, schemas …)
│   ├── member1_doa/            # Steps 00–03 + run_all.py
│   ├── member2_enhance/        # Steps 00–03 + run_all.py
│   ├── member3_analysis/       # Steps 00–04 + run_all.py
│   └── member4_fusion/         # Steps 00–04 + run_all.py
│
├── tests/                      # Smoke tests (pytest)
├── scripts/                    # Shell helpers (setup, run members)
└── docs/                       # Documentation
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repo-url>
cd Audio_explorers_software
```

### 2. Create and activate a virtual environment

```bash
# Option A – use the helper script
bash scripts/setup_env.sh
source .venv/bin/activate

# Option B – manual
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install -r requirements-dev.txt
```

### 3. Place data files

Copy `example_mixture.wav` and `mixture.wav` into `data/`.  
These files are **not** stored in Git.

### 4. Verify the setup

```bash
pytest tests/ -v
```

---

## Data Files

| File | Location | Description |
|---|---|---|
| `example_mixture.wav` | `data/` | Short 4-ch example for quick iteration |
| `mixture.wav` | `data/` | Full 4-ch hearing-aid recording |

Both are 4-channel, 44 100 Hz.  Channel order: **LF, LR, RF, RR**.

---

## Pipeline Overview & Member Responsibilities

### Member 1 – Calibration & DoA

Scripts: `src/member1_doa/00_load_and_stft.py` → `03_track_and_cluster.py`

- Loads WAV files and verifies shape / channel order.
- Computes multi-channel STFTs → `outputs/intermediate/`.
- Calibrates mic-pair time delays (GCC-PHAT) → `outputs/calib/calibration.json`.
- Estimates DoA posterior heatmap → `outputs/doa/doa_posteriors.npy`.
- Clusters DoA tracks into candidates → `outputs/doa/doa_tracks.json`.

### Member 2 – Enhancement / Separation

Scripts: `src/member2_enhance/00_wpe_optional.py` → `03_postfilter_and_export.py`

- Optionally applies WPE dereverberation.
- Builds DoA-guided TF masks from candidate tracks.
- Estimates covariance matrices and steering vectors.
- Applies MVDR beamforming per candidate.
- Post-filters and exports enhanced WAVs → `outputs/separated/spkXX_enhanced.wav`.

### Member 3 – Per-Talker Analysis

Scripts: `src/member3_analysis/00_vad_and_turns.py` → `04_pack_results.py`

- Runs VAD on each candidate → speech segments.
- Identifies language (SpeechBrain).
- Transcribes speech (faster-whisper).
- Estimates pitch / voice type.
- Packs results → `outputs/analysis/spkXX_analysis.json`.

### Member 4 – Fusion & Final Outputs

Scripts: `src/member4_fusion/00_project_structure_and_config.py` → `04_run_end_to_end.py`

- Merges candidate data from all members.
- Scores each candidate (DoA stability, speech duration, language, SNR).
- Selects talker of interest → `outputs/final/talker_of_interest.wav`.
- Generates report figures → `outputs/report/figures/`.
- Writes final scene summary → `outputs/final/final_scene_summary.json`.

---

## Running the Pipeline

```bash
# Individual members
python -m src.member1_doa.run_all
python -m src.member2_enhance.run_all
python -m src.member3_analysis.run_all
python -m src.member4_fusion.run_all

# Full end-to-end
python -m src.member4_fusion.04_run_end_to_end

# Or via shell scripts
bash scripts/run_member1.sh
bash scripts/run_all.sh

# Or via Make
make member1
make all
```

Each member's `run_all.py` will produce **placeholder outputs** if the
real algorithms are not yet implemented, so downstream members can always
test their code.

---

## Output Hand-off Between Members

```
Member 1 produces:
  outputs/calib/calibration.json
  outputs/doa/doa_posteriors.npy
  outputs/doa/doa_tracks.json         ← consumed by Member 2 & 4
  outputs/intermediate/*_stft.npy     ← consumed by Member 2

Member 2 produces:
  outputs/separated/spkXX_enhanced.wav ← consumed by Member 3 & 4
  outputs/separated/spkXX_debug.npz

Member 3 produces:
  outputs/analysis/spkXX_analysis.json ← consumed by Member 4
  outputs/analysis/spkXX_transcript.txt

Member 4 produces:
  outputs/final/talker_of_interest.wav
  outputs/final/final_scene_summary.json
  outputs/report/figures/*.png
```

All intermediate formats are documented in `src/common/json_schema.py`.

---

## Smoke Tests

```bash
# Run all tests
pytest tests/ -v

# Tests included:
#   test_imports.py      – every src module imports without errors
#   test_paths.py        – path constants are valid
#   test_config.py       – config.yaml loads correctly
#   test_json_schema.py  – JSON schema round-trip & validation
```

---

## What Is Implemented vs. Placeholder

| Component | Status |
|---|---|
| Folder structure, config, paths | ✅ Implemented |
| I/O utilities (load/save WAV) | ✅ Implemented |
| STFT / iSTFT wrappers | ✅ Implemented |
| JSON schema & serialisation | ✅ Implemented |
| Plotting helpers | ✅ Implemented |
| Logging utilities | ✅ Implemented |
| GCC-PHAT / calibration | ⬜ Placeholder (TODO) |
| DoA estimation & tracking | ⬜ Placeholder (TODO) |
| WPE dereverberation | ⬜ Placeholder (TODO) |
| MVDR beamforming | ⬜ Placeholder (TODO) |
| VAD / language ID / ASR / pitch | ⬜ Placeholder (TODO) |
| Candidate scoring & fusion | ⬜ Placeholder (TODO) |
| Report figures | ⬜ Placeholder (TODO) |

Every placeholder function logs a warning and returns mock-safe empty
structures so the full pipeline can run end-to-end without crashing.

---

## Git Branching & Collaboration

### Branch naming

| Branch | Owner |
|---|---|
| `main` | Protected – merge via PR only |
| `feature/member1-doa` | Member 1 |
| `feature/member2-enhance` | Member 2 |
| `feature/member3-analysis` | Member 3 |
| `feature/member4-fusion` | Member 4 |

### Workflow

1. Pull latest `main`.
2. Branch off: `git checkout -b feature/member1-doa`.
3. Implement your steps.
4. Run `pytest tests/ -v` and `make lint` before pushing.
5. Open a Pull Request into `main`.
6. At least one teammate reviews before merging.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full PR checklist.
