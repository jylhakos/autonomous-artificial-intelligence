# Scripts

This directory contains Python scripts and shell utilities for vibe coding with Foundry Local and virtual environment management.

## Overview

All executable scripts are organized here to maintain project structure and facilitate automated workflows. Scripts leverage locally-running AI models via Microsoft Foundry Local SDK for privacy-preserving development assistance.

## Setup Requirements

Before running scripts, ensure your environment is properly configured:

### 1. Virtual Environment Setup

Follow the guide: [../VENV_SETUP.md](../VENV_SETUP.md)

**Quick setup:**

```bash
# Navigate to project root
cd ..

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Foundry Local Configuration

Follow the step-by-step Linux setup guide: [../FOUNDRY_LOCAL_SETUP.md](../FOUNDRY_LOCAL_SETUP.md)

**Key requirements:**

- Visual Studio Code with Foundry Toolkit extension
- Foundry Local running (localhost:8080)
- Downloaded AI model (e.g., qwen2.5-0.5b)
- Python 3.10+ with OpenAI SDK

### 3. Project Structure Reference

For complete project organization and architecture: [../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)

## Scripts

### Main Application: prompt_assistant.py

**Purpose**: Interactive tool for generating optimized vibe coding prompts using locally-running AI models via Foundry Local.

**Features:**

- Intent analysis with automatic categorization (8 categories)
- Structured prompt template generation
- Interactive conversational refinement
- Command-line and interactive modes
- Complete data privacy (all processing on-device)
- OpenAI-compatible API integration

**Requirements:**

- Foundry Local running on localhost:8080
- Virtual environment activated
- Python packages: openai, requests, pydantic

**Usage:**

**Interactive Mode:**

```bash
# Ensure virtual environment is active
source ../venv/bin/activate

# Run the assistant
python prompt_assistant.py
```

Interactive session example:

```
💬 Your request: Create a REST API for user authentication with JWT tokens

🔍 Analyzing your request with Foundry Local...
✓ Detected category: code_generation
✓ Intent: Build REST API with JWT authentication

📝 Generating optimized vibe coding prompt...

✨ GENERATED VIBE CODING PROMPT:
======================================================================
[Complete, structured prompt with context, constraints, requirements]
======================================================================

💡 Tip: You can type 'refine' to improve this prompt
```

**Command-Line Mode:**

```bash
python prompt_assistant.py "Create a data pipeline that processes CSV files and generates visualizations"
```

**Environment Variables:**

- `FOUNDRY_MODEL`: Model name (default: qwen2.5-0.5b)
  ```bash
  FOUNDRY_MODEL=phi-3-mini python prompt_assistant.py
  ```

**Integration with VS Code:**

1. Run prompt assistant to generate optimized prompt
2. Copy generated prompt
3. Open GitHub Copilot Chat in VS Code (`Ctrl+Alt+I`)
4. Paste prompt and let AI generate implementation
5. Test and refine iteratively

**Troubleshooting:**

- **Cannot connect to Foundry Local**: Ensure Foundry Local is running

  ```bash
  # Check status
  curl http://localhost:8080/health

  # Start Foundry Local (if using Docker)
  docker run -p 8080:8080 mcr.microsoft.com/foundry-local:latest
  ```

- **Module not found**: Activate virtual environment
  ```bash
  source ../venv/bin/activate
  pip install -r ../requirements.txt
  ```

### Virtual Environment Verification: check_venv.py

**Purpose**: Python script for programmatic verification of virtual environment activation status.

**Usage:**

```bash
python check_venv.py
```

**Output when active:**

```
✓ Virtual environment is active
  VIRTUAL_ENV: /home/laptop/.../pipelines/venv
  Python: 3.12.3
  Location: /home/laptop/.../pipelines/venv/bin/python
```

**Output when not active:**

```
✗ Virtual environment is NOT active
  Run: source venv/bin/activate
```

**Use Cases:**

- Automated build scripts requiring venv validation
- CI/CD pipeline prerequisites
- Development environment verification

### Virtual Environment Verification: check_venv.sh

**Purpose**: Bash script for shell-based verification of virtual environment activation.

**Usage:**

```bash
./check_venv.sh
```

**Output:**

```
✓ Virtual environment is active
  Python version: 3.12.3
  Pip version: 24.0
```

**Use Cases:**

- Shell script integration
- Quick terminal-based verification
- Automated deployment scripts

### Example Application: example_script.py

**Purpose**: Demonstration script showing virtual environment usage with external libraries.

**Features:**

- HTTP connectivity check using requests library
- Python version and executable path display
- Error handling demonstration

**Usage:**

```bash
python example_script.py
```

**Output:**

```
Python version: 3.12.3
Python executable: /home/laptop/.../venv/bin/python
Internet connection: OK (200)
```

**Use Cases:**

- Virtual environment verification template
- HTTP request example
- External library integration demonstration

## Running Scripts in Virtual Environment

### Step-by-Step Execution

**Step 1: Navigate to project root**

```bash
cd /home/laptop/EXERCISES/AUTONOMOUS/autonomous-artificial-intelligence/pipelines
```

**Step 2: Verify virtual environment exists**

```bash
ls -la venv/
# Should show bin/, lib/, include/ directories
```

**Step 3: Activate virtual environment**

```bash
source venv/bin/activate
# Prompt should show (venv) prefix
```

**Step 4: Verify activation**

```bash
# Option 1: Use Python verification
python scripts/check_venv.py

# Option 2: Use bash verification
./scripts/check_venv.sh

# Option 3: Manual check
which python
# Should output: .../pipelines/venv/bin/python
```

**Step 5: Install/update dependencies (if needed)**

```bash
pip install -r requirements.txt
```

**Step 6: Run desired script**

```bash
# Main application
python scripts/prompt_assistant.py

# Example script
python scripts/example_script.py
```

**Step 7: Deactivate when finished**

```bash
deactivate
```

### VS Code Integrated Terminal

If using VS Code integrated terminal, the workspace settings automatically activate the virtual environment:

**Open integrated terminal:**

- Press `` Ctrl+` `` or View → Terminal

**Verify auto-activation:**

```bash
# Prompt should show (venv) prefix
echo $VIRTUAL_ENV
# Should output: /home/laptop/.../pipelines/venv
```

**Run scripts directly:**

```bash
python scripts/prompt_assistant.py
```

### Debugging in VS Code

**Launch Configuration** (already configured in `.vscode/launch.json`):

1. Open script in VS Code editor
2. Set breakpoints by clicking left margin
3. Press `F5` or Run → Start Debugging
4. Select "Python: Prompt Assistant" configuration

**Debug Console:**

- View variables, evaluate expressions
- Step through code execution
- Inspect call stack

## Common Issues and Solutions

### Issue: Virtual Environment Not Activating

**Symptoms:**

- No `(venv)` prefix in prompt
- `which python` points to system Python
- Import errors for installed packages

**Solution:**

```bash
# Deactivate any active environment
deactivate

# Navigate to project root
cd /home/laptop/EXERCISES/AUTONOMOUS/autonomous-artificial-intelligence/pipelines

# Activate with full path
source venv/bin/activate

# Verify
python scripts/check_venv.py
```

### Issue: Module Import Errors

**Symptoms:**

```
ModuleNotFoundError: No module named 'openai'
```

**Solution:**

```bash
# Ensure venv is active
source ../venv/bin/activate

# Install/reinstall dependencies
pip install -r ../requirements.txt

# Verify installation
pip list | grep openai
```

### Issue: Script Permission Denied

**Symptoms:**

```
bash: ./check_venv.sh: Permission denied
```

**Solution:**

```bash
# Make scripts executable
chmod +x check_venv.sh
chmod +x prompt_assistant.py
chmod +x example_script.py

# Or run with python explicitly
python check_venv.sh  # Won't work - use bash
bash check_venv.sh    # Correct
```

### Issue: Foundry Local Connection Failed

**Symptoms:**

```
Error generating completion: ...
Cannot connect to Foundry Local
```

**Solution:**

```bash
# Check if Foundry Local is running
curl http://localhost:8080/health

# If not running, start it
# For Docker:
docker run -p 8080:8080 mcr.microsoft.com/foundry-local:latest

# For native installation (macOS/Windows):
foundry model run qwen2.5-0.5b

# Verify connection
python -c "import requests; print(requests.get('http://localhost:8080/health').status_code)"
```

### Issue: Wrong Python Interpreter in VS Code

**Symptoms:**

- VS Code uses system Python instead of venv
- Imports work in terminal but not in VS Code editor

**Solution:**

1. Press `Ctrl+Shift+P` to open Command Palette
2. Type "Python: Select Interpreter"
3. Choose interpreter from `./venv/bin/python`
4. Reload VS Code window if needed

## Development Workflow

### Typical Vibe Coding Session

1. **Activate environment:**

   ```bash
   cd /home/laptop/EXERCISES/AUTONOMOUS/autonomous-artificial-intelligence/pipelines
   source venv/bin/activate
   ```

2. **Verify setup:**

   ```bash
   python scripts/check_venv.py
   curl http://localhost:8080/health
   ```

3. **Generate prompt:**

   ```bash
   python scripts/prompt_assistant.py
   ```

4. **Describe requirement:**

   ```
   💬 Your request: Create a Flask API with PostgreSQL database
   ```

5. **Copy generated prompt to VS Code/Copilot**

6. **Iterate on implementation:**
   - Test generated code
   - Provide feedback to Copilot
   - Refine until complete

7. **Deactivate when done:**
   ```bash
   deactivate
   ```

### Adding New Scripts

When creating new scripts in this directory:

1. **Add shebang line:**

   ```python
   #!/usr/bin/env python3
   ```

2. **Make executable:**

   ```bash
   chmod +x your_script.py
   ```

3. **Import from project root:**

   ```python
   import sys
   sys.path.append('..')
   ```

4. **Document in this README:**
   - Purpose and features
   - Usage examples
   - Dependencies
   - Troubleshooting

5. **Update PROJECT_STRUCTURE.md** with new script details

## Best Practices

### Virtual Environment

- **Always activate** before running scripts
- **Verify activation** using check_venv scripts
- **Keep requirements.txt updated** with `pip freeze > requirements.txt`
- **Don't commit venv/** to version control (already in .gitignore)

### Script Development

- **Use relative imports** for project modules
- **Handle errors gracefully** with try-except blocks
- **Log operations** for debugging and auditing
- **Document functions** with docstrings
- **Follow PEP 8** style guidelines

### Foundry Local Usage

- **Check connection** before making API calls
- **Handle timeouts** appropriately
- **Use appropriate model** for task complexity
- **Monitor resource usage** (CPU/GPU/NPU)
- **Log API interactions** for debugging

### Security

- **Review AI-generated code** before deployment
- **Don't commit API keys** (not needed for local Foundry)
- **Validate user inputs** to prevent injection attacks
- **Keep dependencies updated** for security patches

## Additional Resources

- **Main Documentation**: [../README.md](../README.md) - SDLC and Vibe Coding guide
- **Project Structure**: [../PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) - Architecture and file organization
- **Virtual Environment Setup**: [../VENV_SETUP.md](../VENV_SETUP.md) - Detailed venv configuration
- **Foundry Local Setup**: [../FOUNDRY_LOCAL_SETUP.md](../FOUNDRY_LOCAL_SETUP.md) - Complete Linux installation guide
- **Requirements**: [../requirements.txt](../requirements.txt) - Python package dependencies
- **Git Ignore**: [../.gitignore](../.gitignore) - Version control exclusions

## Quick Reference

### Essential Commands

```bash
# Activate virtual environment
source ../venv/bin/activate

# Run prompt assistant (interactive)
python prompt_assistant.py

# Run prompt assistant (CLI)
python prompt_assistant.py "Your request here"

# Check venv status
python check_venv.py

# Run example
python example_script.py

# Install dependencies
pip install -r ../requirements.txt

# Deactivate venv
deactivate
```

### File Locations

- **Scripts**: `pipelines/scripts/`
- **Virtual Environment**: `pipelines/venv/`
- **Dependencies**: `pipelines/requirements.txt`
- **Documentation**: `pipelines/*.md`
- **VS Code Config**: `pipelines/.vscode/`

---

**Project**: Vibe Coding with SDLC

**Last Updated**: April 23, 2026
