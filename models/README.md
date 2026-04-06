# models/

This directory stores pre-trained model weights and caches.  
**Do NOT commit weights to Git** — they are listed in `.gitignore`.

## Sub-directories

| Directory | Purpose |
|---|---|
| `whisper/` | Faster-whisper model files (downloaded on first run) |
| `speechbrain/` | SpeechBrain LangID / embeddings (downloaded on first run) |
| `optional_separation/` | Optional source-separation model weights |
| `cache/` | Generic cache for any model/library that supports a cache dir |

## How models are obtained

Most models **download automatically** the first time you run the pipeline.
By default they cache in `~/.cache/` (HuggingFace / SpeechBrain default).

To keep everything inside this repo instead, set environment variables:

```bash
export HF_HOME=./models/cache
export WHISPER_CACHE_DIR=./models/whisper
export SPEECHBRAIN_CACHE=./models/speechbrain
```

Or copy these lines into your `.env` file (see `.env.example` at the root).

## Approximate sizes

| Model | Size |
|---|---|
| faster-whisper large-v3 | ~3 GB |
| SpeechBrain lang-id-voxlingua107-ecapa | ~100 MB |
| Optional separation models | varies |

Make sure you have enough disk space before running the pipeline.
