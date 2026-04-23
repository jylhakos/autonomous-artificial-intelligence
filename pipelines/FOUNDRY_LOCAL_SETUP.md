# Setting Up Vibe Coding Environment with Foundry Local on Linux

This tutorial provides step-by-step instructions for configuring a vibe coding environment using Visual Studio Code, terminal, and Microsoft Foundry Local on Linux (Debian).

## Overview

**Foundry Local** is Microsoft's end-to-end local AI solution that enables you to run AI models entirely on your device (NPU, GPU, or CPU) without cloud dependencies. This setup enables true vibe coding where AI agents build entire applications through natural language conversation while maintaining data privacy and offline functionality.

## Prerequisites

- **Operating System**: Ubuntu 20.04+ or compatible Linux distribution
- **Hardware**: Modern CPU (NPU/GPU optional for acceleration)
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Storage**: At least 10GB free space for models
- **Python**: Version 3.10 or higher
- **VS Code**: Latest version

## Step 1: Install Visual Studio Code

If not already installed:

```bash
# Download and install VS Code
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list > /dev/null
rm -f packages.microsoft.gpg

sudo apt update
sudo apt install code
```

Launch VS Code:

```bash
code
```

## Step 2: Install Foundry Toolkit for VS Code

The Foundry Toolkit extension provides integration with Foundry Local directly in VS Code.

### Installation Options

**Option A: Install from Marketplace**

1. Open VS Code
2. Click Extensions icon (or press `Ctrl+Shift+X`)
3. Search for "Foundry" or "Microsoft Foundry"
4. Click **Install** on "Foundry for Visual Studio Code" extension

**Option B: Install from Command Palette**

1. Press `F1` to open Command Palette
2. Type: `Extensions: Install Extensions`
3. Search for "Foundry"
4. Select and install

**Option C: Install via Command Line**

```bash
code --install-extension TeamsDevApp.vscode-ai-foundry
```

After installation, the Foundry icon appears in the left navigation bar.

## Step 3: Install Foundry Local SDK

Since Foundry Local primarily supports macOS/Windows CLI natively, on Linux we'll use the SDK approach through Python.

### Install Python SDK

```bash
# Ensure you have Python 3.10+
python3 --version

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Foundry Local SDK and dependencies
pip install --upgrade pip
pip install openai requests
```

### Alternative: Use Docker (Recommended for Linux)

If available, Foundry Local can be run via Docker for better Linux compatibility:

```bash
# Pull Foundry Local container (if available)
docker pull mcr.microsoft.com/foundry-local:latest

# Run Foundry Local server
docker run -p 8080:8080 mcr.microsoft.com/foundry-local:latest
```

### Verify Installation

Check if Foundry Local SDK is accessible:

```python
# test_foundry.py
import requests

try:
    response = requests.get("http://localhost:8080/health", timeout=2)
    if response.status_code == 200:
        print("✓ Foundry Local is running")
    else:
        print("⚠ Foundry Local responded but not healthy")
except Exception as e:
    print(f"✗ Cannot connect to Foundry Local: {e}")
```

Run:

```bash
python test_foundry.py
```

## Step 4: Download AI Models

Foundry Local requires downloading models to run locally.

### Available Models

Common models for vibe coding:

- **qwen2.5-0.5b**: Lightweight, fast (0.5B parameters)
- **qwen2.5-1.5b**: Balanced performance (1.5B parameters)
- **phi-3-mini**: Microsoft's efficient model (3.8B parameters)
- **llama-3.2-1b**: Meta's compact model (1B parameters)

### Download and Run Model

```bash
# If using foundry CLI (macOS/Windows native):
foundry model run qwen2.5-0.5b

# For Linux SDK approach, models are managed through API calls
# The first API request will trigger model download
```

## Step 5: Configure VS Code Workspace

### Set Up Project Structure

```bash
# Navigate to your workspace
cd ~/EXERCISES/AUTONOMOUS/autonomous-artificial-intelligence/pipelines

# Verify virtual environment
source venv/bin/activate

# Ensure dependencies installed
pip install -r requirements.txt
```

### Configure VS Code Settings

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.analysis.autoImportCompletions": true,
  "files.autoSave": "afterDelay",
  "editor.formatOnSave": true,
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true
}
```

### Configure Launch Configuration

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Prompt Assistant",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/prompt_assistant.py",
      "console": "integratedTerminal",
      "env": {
        "FOUNDRY_MODEL": "qwen2.5-0.5b"
      }
    }
  ]
}
```

## Step 6: Run the Prompt Assistant

### Terminal Usage

**Interactive Mode:**

```bash
python prompt_assistant.py
```

**Single Request Mode:**

```bash
python prompt_assistant.py "Create a REST API for user authentication with JWT tokens"
```

### VS Code Integrated Terminal

1. Open VS Code: `code .`
2. Open integrated terminal: `` Ctrl+` ``
3. Activate virtual environment (if not auto-activated):
   ```bash
   source venv/bin/activate
   ```
4. Run the assistant:
   ```bash
   python prompt_assistant.py
   ```

### Debugging in VS Code

1. Open `prompt_assistant.py` in editor
2. Set breakpoints (click left margin)
3. Press `F5` to start debugging
4. Use Debug Console to inspect variables

## Step 7: Verify Vibe Coding Setup

Test the complete workflow:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Verify Python and packages
python --version
pip list | grep openai

# 3. Test Foundry Local connection
python -c "import requests; print('✓ Connected' if requests.get('http://localhost:8080/health', timeout=2).status_code == 200 else '✗ Failed')"

# 4. Run prompt assistant
python prompt_assistant.py
```

Expected output:

```
🔧 Initializing Foundry Local connection...
✓ Successfully connected to Foundry Local

🚀 Vibe Coding Prompt Assistant with Foundry Local
======================================================================
This tool helps you create effective prompts for AI-assisted development.
It runs entirely on your local hardware using Foundry Local.
```

## Step 8: Using the Prompt Assistant for Vibe Coding

### Example Workflow

1. **Launch the assistant:**

   ```bash
   python prompt_assistant.py
   ```

2. **Describe what you want to build:**

   ```
   💬 Your request: Create a Python web scraper that extracts product prices from e-commerce websites and saves them to a CSV file
   ```

3. **Review generated prompt:**
   The assistant analyzes your request and generates an optimized vibe coding prompt with:
   - Clear task description
   - Technical context
   - Constraints and requirements
   - Expected output format

4. **Refine if needed:**

   ```
   💬 Your request: refine
   How would you like to refine the prompt? Add error handling for network timeouts and rate limiting
   ```

5. **Copy prompt to VS Code:**
   - Copy the generated prompt
   - Open GitHub Copilot Chat in VS Code (`Ctrl+Alt+I`)
   - Paste the prompt
   - Let AI generate the implementation

6. **Iterate conversationally:**
   - Test the generated code
   - Provide feedback to Copilot
   - Refine until complete

## Troubleshooting

### Foundry Local Not Starting

**Problem:** Cannot connect to Foundry Local

**Solutions:**

1. Check if service is running:

   ```bash
   curl http://localhost:8080/health
   ```

2. Verify port 8080 is not in use:

   ```bash
   sudo netstat -tlnp | grep 8080
   ```

3. Check Docker container status (if using Docker):
   ```bash
   docker ps | grep foundry
   docker logs <container-id>
   ```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'openai'`

**Solution:**

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Virtual Environment Not Activating

**Problem:** Commands use system Python instead of venv

**Solution:**

```bash
deactivate  # if in wrong environment
source venv/bin/activate
which python  # should show venv/bin/python
```

### Model Download Fails

**Problem:** Model fails to download or load

**Solutions:**

1. Check internet connectivity
2. Verify sufficient disk space:
   ```bash
   df -h
   ```
3. Try smaller model (e.g., qwen2.5-0.5b instead of larger models)

## Best Practices

### Performance Optimization

1. **Use appropriate model size:**
   - Small projects: qwen2.5-0.5b (fastest)
   - Medium projects: phi-3-mini (balanced)
   - Large projects: larger models if hardware supports

2. **Hardware acceleration:**
   - Check if GPU available: `nvidia-smi`
   - Configure Foundry Local to use GPU if available

3. **Batch operations:**
   - Generate multiple prompts in one session
   - Minimize model load/unload cycles

### Security Considerations

1. **Data privacy:** All processing happens locally—no data sent to cloud
2. **Model provenance:** Only use models from trusted sources
3. **Code review:** Always review AI-generated code before deployment
4. **Access control:** Secure your local Foundry Local endpoint

### Workflow Integration

1. **AGENTS.md configuration:**
   - Create `AGENTS.md` in project root with guidelines
   - Improves prompt assistant and Copilot outputs

2. **Version control:**
   - Commit generated prompts for team collaboration
   - Track prompt evolution with git

3. **Documentation:**
   - Document effective prompts that work well
   - Share successful patterns with team

## Additional Resources

- **Foundry Local Documentation:** https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/get-started
- **Foundry Toolkit for VS Code:** https://code.visualstudio.com/docs/intelligentapps/overview
- **GitHub Repository:** https://github.com/microsoft/Foundry-Local
- **Sample Code:** https://github.com/microsoft/Foundry-Local/tree/main/samples/python
- **Model Catalog:** https://www.foundrylocal.ai/models

## Next Steps

1. **Explore samples:** Check out official Foundry Local Python samples
2. **Customize assistant:** Modify `prompt_assistant.py` for your specific needs
3. **Integrate with workflows:** Connect to your SDLC processes
4. **Experiment with models:** Try different models for different tasks
5. **Build applications:** Use vibe coding to create production applications

---

**Platform:** Linux (Ubuntu/Debian)

**Last Updated:** April 23, 2026
