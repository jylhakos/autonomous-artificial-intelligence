#!/usr/bin/env python3
"""
Virtual Environment Verification Script
Checks if running inside a Python virtual environment
"""

import sys
import os

def check_virtual_env():
    """Check if we're running in a virtual environment."""
    print("=" * 50)
    print("Python Virtual Environment Status")
    print("=" * 50)
    
    # Show Python executable location
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    
    # Check if we're in a virtual environment
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    print(f"Virtual environment active: {in_venv}")
    
    # Show VIRTUAL_ENV environment variable if set
    venv_path = os.environ.get('VIRTUAL_ENV', 'Not set')
    print(f"VIRTUAL_ENV path: {venv_path}")
    
    print("=" * 50)
    
    if in_venv:
        print("✓ Virtual environment is ACTIVE")
    else:
        print("✗ Virtual environment is NOT active")
        print("  Run: source venv/bin/activate")
    
    return in_venv

if __name__ == "__main__":
    check_virtual_env()
