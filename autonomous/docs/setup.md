# Setup

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment tool (venv)
- VS Code (recommended)

### Installation Steps

#### 1. Clone or Download the Project

```bash
cd /path/to/autonomous-artificial-intelligence/autonomous
```

#### 2. Create Virtual Environment

**On Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Upgrade pip

```bash
pip install --upgrade pip
```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google
GOOGLE_API_KEY=your_google_api_key_here

# LiteLLM (optional)
LITELLM_LOG=DEBUG
```

#### 6. Verify Installation

```bash
python examples/simple_agent.py
```

## VS Code Setup

### 1. Install VS Code Extensions

- Python
- Pylance
- GitHub Copilot (optional)
- GitHub Copilot Chat (optional)

### 2. Configure Python Interpreter

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `./venv/bin/python`

### 3. Configure VS Code Settings

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false
}
```

## Running Examples

### Simple Agent Example

```bash
python examples/simple_agent.py
```

### Multi-Agent System

```bash
python examples/multi_agent_system.py
```

### Multi-Model Chat (Streamlit)

```bash
streamlit run examples/multi_model_chat.py
```

## Development Workflow

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
```

### Linting

```bash
flake8 .
```

### Type Checking

```bash
mypy agents/ tools/ workflows/
```

## Troubleshooting

### Issue: Module not found

**Solution:**

```bash
pip install -r requirements.txt
```

### Issue: API key errors

**Solution:**
Ensure `.env` file is created and API keys are correct

### Issue: Virtual environment not activating

**Solution:**

- On Linux/macOS: `source venv/bin/activate`
- On Windows: `venv\Scripts\activate`

### Issue: Permission denied on Linux

**Solution:**

```bash
chmod +x venv/bin/activate
```

## Next Steps

1. Read the [Architecture Guide](architecture.md)
2. Follow the [Tutorials](tutorials.md)
3. Explore example code in `examples/`

## Resources

- [Main README](../README.md)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [VS Code Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial)
