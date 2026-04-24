#!/bin/bash
# Virtual Environment Verification Script (Bash version)

check_venv() {
    echo "=================================================="
    echo "Virtual Environment Status (Bash)"
    echo "=================================================="
    
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "✓ Virtual environment is ACTIVE"
        echo "  Path: $VIRTUAL_ENV"
        echo "  Python: $(which python)"
        echo "  Version: $(python --version)"
        echo "  Pip: $(which pip)"
    else
        echo "✗ Virtual environment is NOT active"
        echo "  Run: source venv/bin/activate"
    fi
    
    echo "=================================================="
}

check_venv
