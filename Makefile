# ============================================================================
# Makefile – Audio Explorers convenience targets
# ============================================================================
# Usage:
#   make setup        – create venv & install deps
#   make test         – run pytest
#   make lint         – run ruff + black --check
#   make member1      – run Member 1 pipeline
#   make member2      – run Member 2 pipeline
#   make member3      – run Member 3 pipeline
#   make member4      – run Member 4 pipeline
#   make all          – run full end-to-end pipeline
#   make clean        – remove generated outputs
# ============================================================================

PYTHON ?= python3
VENV   := .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

.PHONY: setup test lint member1 member2 member3 member4 all clean help

help:
	@echo "Available targets:"
	@echo "  setup    – create venv and install deps"
	@echo "  test     – run pytest"
	@echo "  lint     – run ruff + black --check"
	@echo "  member1  – run Member 1 (DoA) pipeline"
	@echo "  member2  – run Member 2 (enhancement) pipeline"
	@echo "  member3  – run Member 3 (analysis) pipeline"
	@echo "  member4  – run Member 4 (fusion) pipeline"
	@echo "  all      – run full end-to-end pipeline"
	@echo "  clean    – remove generated outputs"

# ── Environment setup ──────────────────────────────────────────────────
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PIP) install -r requirements-dev.txt
	@echo "\n✓ Environment ready. Activate with: source $(VENV)/bin/activate"

# ── Quality ────────────────────────────────────────────────────────────
test:
	$(PY) -m pytest tests/ -v --tb=short

lint:
	$(PY) -m ruff check src/ tests/
	$(PY) -m black --check src/ tests/

# ── Member pipelines ──────────────────────────────────────────────────
member1:
	$(PY) -m src.member1_doa.run_all

member2:
	$(PY) -m src.member2_enhance.run_all

member3:
	$(PY) -m src.member3_analysis.run_all

member4:
	$(PY) -m src.member4_fusion.run_all

all: member1 member2 member3 member4
	@echo "\n✓ Full pipeline complete."

# ── Cleanup ────────────────────────────────────────────────────────────
clean:
	rm -f outputs/calib/*.json
	rm -f outputs/doa/*.npy outputs/doa/*.json
	rm -f outputs/intermediate/*.npy
	rm -f outputs/separated/*.wav outputs/separated/*.npz
	rm -f outputs/analysis/*.json outputs/analysis/*.txt
	rm -f outputs/final/*.wav outputs/final/*.json
	rm -rf outputs/final/figures/*.png
	rm -f outputs/report/figures/*.png outputs/report/tables/*.csv
	@echo "✓ Generated outputs removed."
