#!/usr/bin/env bash
# ============================================================================
# run_member3.sh – Run the Member 3 (Analysis) pipeline.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "Running Member 3 – Analysis pipeline …"
python -m src.member3_analysis.run_all "$@"
