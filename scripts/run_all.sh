#!/usr/bin/env bash
# ============================================================================
# run_all.sh – Run the full end-to-end pipeline (all 4 members).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "Running full end-to-end pipeline …"
python -m src.member4_fusion.04_run_end_to_end "$@"
