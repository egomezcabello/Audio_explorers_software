#!/usr/bin/env bash
# ============================================================================
# run_member4.sh – Run the Member 4 (Fusion) pipeline.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
echo "Running Member 4 – Fusion pipeline …"
python -m src.member4_fusion.run_all "$@"
