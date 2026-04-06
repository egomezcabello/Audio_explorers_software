#!/usr/bin/env bash
# ============================================================================
# setup_env.sh – Create virtual environment and install all dependencies.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

echo "──── Audio Explorers – Environment Setup ────"
echo "Project root : $PROJECT_ROOT"
echo "Virtual env  : $VENV_DIR"
echo ""

# Create venv if missing
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment …"
    python3 -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install project in editable mode with dev extras
pip install -e "$PROJECT_ROOT[dev]"
pip install -r "$PROJECT_ROOT/requirements-dev.txt"

echo ""
echo "✓ Setup complete.  Activate with:"
echo "    source $VENV_DIR/bin/activate"
