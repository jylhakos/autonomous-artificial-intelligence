#!/usr/bin/env bash
# Creates and populates a virtual environment for the service load forecasting scripts.
#
# After running this script activate the environment in your shell:
#   source venv/bin/activate
#
# Usage:
#   cd scripts/forecasting
#   chmod +x setup_venv_forecast.sh
#   ./setup_venv_forecast.sh
#   source venv/bin/activate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "Creating virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Upgrading pip ..."
"$VENV_DIR/bin/pip" install --upgrade pip

echo "Installing forecast dependencies ..."
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements_forecast.txt"

echo ""
echo "Setup complete."
echo "Activate the forecasting environment with:"
echo "  source $VENV_DIR/bin/activate"
