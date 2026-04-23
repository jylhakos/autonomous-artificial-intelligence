# Python Virtual Environment Setup Guide

## Virtual Environment Setup

The Python virtual environment has been successfully created and configured for this project.

## Quick Start

### Activate the Virtual Environment

```bash
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal prompt when active.

### Deactivate the Virtual Environment

```bash
deactivate
```

## Verify Virtual Environment Status

Two scripts are provided to check if the virtual environment is active:

### Bash Script
```bash
./check_venv.sh
```

### Python Script
```bash
python check_venv.py
```

## Installing Libraries

Always ensure the virtual environment is active before installing libraries:

```bash
# Check if venv is active
./check_venv.sh

# Install a single library
pip install numpy

# Install multiple libraries
pip install pandas matplotlib requests

# Install from requirements file
pip install -r requirements.txt

# Upgrade a library
pip install --upgrade numpy

# List installed packages
pip list

# Generate requirements file from installed packages
pip freeze > requirements.txt
```

## Running Scripts

Before running any Python script, verify the virtual environment is active:

```bash
# Check venv status
./check_venv.sh

# Run a Python script
python your_script.py

# Run with arguments
python your_script.py --arg1 value1 --arg2 value2
```

## Bash Terminal Commands

Execute bash commands within the virtual environment context:

```bash
# List Python packages
pip list

# Show package information
pip show package_name

# Check Python version
python --version

# Get Python executable path
which python

# Run Python module as script
python -m module_name
```

## Git Configuration

The `.gitignore` file has been configured to exclude:
- Virtual environment directories (`venv/`, `.venv/`, `env/`)
- Python cache files (`__pycache__/`, `*.pyc`)
- Compiled binaries (`*.so`, `*.egg`)
- IDE configuration files (`.vscode/`, `.idea/`)
- Distribution files (`dist/`, `build/`)
- Test coverage reports
- Log files

## Best Practices

1. **Always activate the virtual environment** before working on the project
2. **Keep requirements.txt updated** with all project dependencies
3. **Never commit the venv folder** to version control
4. **Use pip freeze** to capture exact versions of dependencies
5. **Document special installation requirements** in this file
6. **Test scripts in the virtual environment** before committing

## Common Commands

```bash
# Create a new virtual environment
python3 -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black .

# Lint code
flake8 .

# Type check
mypy .
```

## Troubleshooting

### Virtual Environment Not Activating
- Ensure you're in the project root directory
- Check that `venv/bin/activate` exists
- Try recreating: `rm -rf venv && python3 -m venv venv`

### Package Installation Fails
- Upgrade pip: `pip install --upgrade pip`
- Check Python version compatibility
- Verify internet connection

### Wrong Python Version
- Deactivate and reactivate the virtual environment
- Verify with: `python --version`
- Check executable: `which python`

## Environment Variables

Set environment variables within the virtual environment:

```bash
# Temporarily (current session only)
export MY_VAR="value"

# Permanently (add to venv/bin/activate)
echo 'export MY_VAR="value"' >> venv/bin/activate
```

## Project Structure

```
pipelines/
├── venv/                    # Virtual environment (excluded from git)
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
├── VENV_SETUP.md           # This file
├── requirements.txt        # Python dependencies
├── check_venv.sh          # Bash venv check script
└── check_venv.py          # Python venv check script
```
