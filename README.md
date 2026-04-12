# Audio Explorers – Hearing-Aid Microphone Array Scene Analysis

A 4-channel hearing-aid (BTE) microphone array pipeline that localises
multiple talkers, separates them with beamforming, analyses each one
(transcription, pitch, language), and picks the talker of interest.

**Audio format:** 4 channels, 44 100 Hz, WAV.
**Channel order:** `LF · LR · RF · RR`

---

## Installation

### 1. Clone and set up

```bash
git clone <repo-url>
cd Audio_explorers_software

# Option A — helper script (creates .venv, installs everything)
bash scripts/setup_env.sh
source .venv/bin/activate

# Option B — manual
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Core dependencies (numpy, scipy, librosa,
pyroomacoustics, faster-whisper, speechbrain, etc.) are declared in
`pyproject.toml` and installed automatically.

### 2. Models

Two pre-trained models are used:

| Model | Library | Downloaded automatically? |
|---|---|---|
| `Systran/faster-whisper-large-v3` | faster-whisper | Yes, on first run |
| `speechbrain/lang-id-voxlingua107-ecapa` | SpeechBrain | Yes, on first run |

By default they cache under `~/.cache/`. To keep them inside the repo set:

```bash
export HF_HOME=models/whisper
export SPEECHBRAIN_CACHE=models/speechbrain
```

### 3. Data

Place `mixture.wav` (and optionally `example_mixture.wav`) in `data/`.
These are 4-channel 44.1 kHz WAV files and are not stored in Git.

### 4. Verify

```bash
pytest tests/ -v
```

---

## Pipeline Members

### Member 1 — Direction of Arrival

`src/member1_doa/` (steps 00–03)

Loads the multichannel WAV, computes STFTs, calibrates microphone-pair
delays with GCC-PHAT, builds an SRP-PHAT azimuth posterior over time,
and clusters the posterior into speaker-direction tracks.

Outputs: `outputs/calib/calibration.json`, `outputs/doa/doa_posteriors.npy`,
`outputs/doa/doa_tracks.json`.

### Member 2 — Enhancement

`src/member2_enhance/` (steps 00–03)

For each candidate direction: builds a DoA-guided time-frequency mask
(competitive soft masking across candidates), estimates spatial covariance
matrices, applies MVDR beamforming with interferer null-steering, and
runs a Wiener post-filter with spatial mask gating.

Outputs: `outputs/separated/spkXX_enhanced.wav`.

### Member 3 — Per-Talker Analysis

`src/member3_analysis/` (steps 00–04)

Runs WebRTC VAD on each enhanced signal, identifies the language with
SpeechBrain, transcribes with faster-whisper (large-v3), and estimates
fundamental frequency for male/female/child classification.

Outputs: `outputs/analysis/spkXX_analysis.json`, `outputs/analysis/spkXX_transcript.txt`.

### Member 4 — Fusion

`src/member4_fusion/` (steps 00–04)

Merges candidates from all members, scores them by DoA stability,
speech duration, language match, and SNR, selects the talker of interest,
copies its WAV, and generates summary figures.

Outputs: `outputs/final/talker_of_interest.wav`, `outputs/final/final_scene_summary.json`,
`outputs/report/figures/`.

---

## Running the Pipeline

All parameters live in `config.yaml` (contains tuned values from recent
parameter optimisation).

```bash
# Run members sequentially (each depends on the previous)
python -m src.member1_doa.run_all
python -m src.member2_enhance.run_all
python -m src.member3_analysis.run_all
python -m src.member4_fusion.run_all

# Or run everything at once
bash scripts/run_all.sh
```

---

## Output Flow

```
M1 → outputs/doa/doa_tracks.json        → M2, M4
      outputs/intermediate/*_stft.npy    → M2

M2 → outputs/separated/spkXX_enhanced.wav → M3, M4

M3 → outputs/analysis/spkXX_analysis.json → M4

M4 → outputs/final/talker_of_interest.wav
     outputs/final/final_scene_summary.json
```

---

## Tests

```bash
pytest tests/ -v
```

Covers: imports, path constants, config loading, JSON schema validation.
