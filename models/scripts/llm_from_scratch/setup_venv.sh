#!/usr/bin/env bash
# Create and populate a Python virtual environment for the LLM scripts.
#
# Usage:
#   chmod +x setup_venv.sh
#   ./setup_venv.sh
#
# After this script completes, activate the environment in your current shell:
#   source venv/bin/activate
#
# To deactivate later:
#   deactivate

set -euo pipefail

VENV_DIR="venv"
REQUIREMENTS="requirements.txt"

if [[ ! -f "$REQUIREMENTS" ]]; then
    echo "Error: $REQUIREMENTS not found. Run this script from the scripts/llm_from_scratch/ directory."
    exit 1
fi

echo "Creating virtual environment in '$VENV_DIR/'..."
python3 -m venv "$VENV_DIR"

echo "Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet

echo "Installing dependencies from $REQUIREMENTS..."
"$VENV_DIR/bin/pip" install -r "$REQUIREMENTS"

echo ""
echo "Virtual environment ready."
echo ""
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then run the scripts:"
echo "  python train.py              # pretrain the base model"
echo "  python text_classifier.py   # fine-tune as text classifier"
echo "  python chatbot.py           # interactive chatbot"
echo ""
echo "Deactivate with:"
echo "  deactivate"
