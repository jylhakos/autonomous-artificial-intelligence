# Setup

This document provides detailed instructions for setting up the Autonomous AI Agent Evaluation framework.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning repositories)

## Installation Steps

### 1. Verify Python Installation

Check your Python version:

```bash
python3 --version
```

You should see Python 3.8 or higher. If not, install Python from [python.org](https://www.python.org/downloads/).

### 2. Navigate to Project Directory

```bash
cd /path/to/evaluation
```

### 3. Create Virtual Environment

**On Linux/macOS:**

```bash
python3 -m venv venv
```

**On Windows:**

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

**On Linux/macOS:**

```bash
source venv/bin/activate
```

**On Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
```

**On Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

**Note:** If you see an error on Windows PowerShell about execution policies, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 5. Upgrade pip

```bash
pip install --upgrade pip
```

### 6. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- LangChain and LangChain integrations
- LangSmith for observability
- Anthropic SDK for Claude models
- Supporting libraries

### 7. Configure Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit the `.env` file with your favorite text editor:

```bash
nano .env
# or
vim .env
# or
code .env  # if using VS Code
```

### 8. Get API Keys

#### LangSmith API Key (Required)

1. Go to [https://smith.langchain.com/](https://smith.langchain.com/)
2. Sign up for a free account
3. Navigate to Settings → API Keys
4. Click "Create API Key"
5. Copy the key and paste it in your `.env` file:

```bash
LANGCHAIN_API_KEY=lsv2_pt_abc123...
```

#### Anthropic API Key (Required)

1. Go to [https://console.anthropic.com/](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new API key
5. Copy the key and paste it in your `.env` file:

```bash
ANTHROPIC_API_KEY=sk-ant-abc123...
```

#### OpenAI API Key (Optional)

If you want to use OpenAI models instead of Anthropic:

1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add to `.env`:

```bash
OPENAI_API_KEY=sk-abc123...
```

### 9. Verify Installation

Run a quick test:

```bash
python -c "import langchain, langsmith; print('Installation successful!')"
```

If you see "Installation successful!", you're ready to go!

## Running the Application

### Using the Quick Start Script (Linux/macOS)

```bash
./quickstart.sh
```

### Manual Method

1. Activate the virtual environment (if not already activated):

```bash
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

2. Run the main application:

```bash
python main.py
```

### Running Individual Examples

You can also run examples directly:

```bash
# Basic agent
python -m src.basic_agent

# Agent with tracing
python -m src.agent_with_tracing

# Agent evaluation
python -m src.agent_evaluation

# Online evaluation
python -m src.online_evaluation
```

## Troubleshooting

### Issue: Module not found

**Solution:** Make sure your virtual environment is activated and dependencies are installed:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: API key not working

**Solution:** 
1. Verify the key is correctly set in `.env`
2. Make sure there are no extra spaces or quotes
3. Check that the `.env` file is in the project root directory
4. Restart your terminal or reload environment variables

### Issue: Permission denied on quickstart.sh

**Solution:** Make the script executable:

```bash
chmod +x quickstart.sh
```

### Issue: Import errors with langchain

**Solution:** Reinstall dependencies:

```bash
pip uninstall langchain langchain-anthropic langsmith
pip install -r requirements.txt
```

### Issue: Tracing not appearing in LangSmith

**Solution:**
1. Verify `LANGCHAIN_TRACING_V2=true` in `.env`
2. Check your API key is valid
3. Ensure `LANGCHAIN_PROJECT` is set
4. Wait a few seconds for traces to appear in the dashboard

## Deactivating Virtual Environment

When you're done working:

```bash
deactivate
```

## Updating Dependencies

To update to the latest versions:

```bash
pip install --upgrade -r requirements.txt
```

## Next Steps

Once setup is complete:

1. Read the [README.md](README.md) for project overview
2. Run `python main.py` to see interactive examples
3. Explore the source code in the `src/` directory
4. Visit [https://smith.langchain.com/](https://smith.langchain.com/) to view your traces
5. Experiment with creating your own agents and evaluations

## Additional Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [LangSmith Documentation](https://docs.langchain.com/langsmith/)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [Agent Observability Guide](https://www.langchain.com/conceptual-guides/agent-observability-powers-agent-evaluation)

## Support

If you encounter issues not covered in this guide:

1. Check the [LangChain Discord](https://discord.gg/langchain)
2. Review [LangSmith Support](https://smith.langchain.com/)
3. Search [Stack Overflow](https://stackoverflow.com/questions/tagged/langchain)
