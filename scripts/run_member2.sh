#!/usr/bin/env bash
# ============================================================================
# run_member2.sh – Run the Member 2 (Enhancement) pipeline.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "Running Member 2 – Enhancement pipeline …"
python -m src.member2_enhance.run_all "$@"
