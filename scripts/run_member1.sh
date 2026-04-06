#!/usr/bin/env bash
# ============================================================================
# run_member1.sh – Run the Member 1 (DoA) pipeline.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "Running Member 1 – DoA pipeline …"
python -m src.member1_doa.run_all "$@"
