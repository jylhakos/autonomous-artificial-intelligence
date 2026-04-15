# Quick Start - Collaboration Folder

This is a reference for getting started with collaborative AI agents.

## API Key Required

**You need to set your OPENAI_API_KEY before running any examples or tests.**

Without this, all code will fail. Get your key from: https://platform.openai.com/api-keys

```bash
# Set API key (choose your platform)
export OPENAI_API_KEY='your-key-here'  # Linux/Mac
set OPENAI_API_KEY=your-key-here  # Windows CMD
$env:OPENAI_API_KEY="your-key-here"  # Windows PowerShell

# Verify it's set
echo $OPENAI_API_KEY  # Should show your key
```

## 1-Minute Setup

```bash
# Navigate to collaboration folder
cd collaboration

# Activate virtual environment (already created)
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Set your API key
export OPENAI_API_KEY='your-openai-api-key-here'

# Run first example
python autogen_hello_agent.py
```

## Files Overview

### Examples (Run These)
- `autogen_hello_agent.py` - Start here, basic single agent
- `autogen_multi_agent_collaboration.py` - Multiple agents working together
- `hybrid_agent_collaboration.py` - Software development team simulation
- `claude_agent_basic.py` - Claude Agent SDK examples (optional)

### Testing (Validate Collaboration)
- `test_code_collaboration.py` - Test code generation and review
- `test_human_feedback.py` - Test human-in-the-loop feedback
- `test_collaboration_suite.py` - Run all tests

### Utilities
- `check_setup.py` - Verify your environment is configured
- `run_all_examples.sh` - Run all examples sequentially
- `requirements.txt` - Python package dependencies
- `README.md` - Complete document

## Status Check

Run this to verify everything is set up:
```bash
python check_setup.py
```

## Common Commands

### Run Examples
```bash
# Basic
python autogen_hello_agent.py

# Multi-agent
python autogen_multi_agent_collaboration.py

# All examples
./run_all_examples.sh
```

### Run Tests
```bash
# Individual tests
python test_code_collaboration.py
python test_human_feedback.py

# All tests
python test_collaboration_suite.py
```

### VS Code
1. Open folder: `code .`
2. Select interpreter: Ctrl+Shift+P > "Python: Select Interpreter" > Choose venv
3. Open terminal: Ctrl+`
4. Run any file with F5 or click Run button

## What You'll Learn

1. How to create AI agents with specific roles
2. How agents communicate and share context
3. How to test agent collaboration
4. How to analyze chat logs and turn-taking
5. How to integrate human feedback
6. How to use VS Code for agent development

## Next Steps

1. Read `README.md` for document
2. Run `autogen_hello_agent.py` to test basic setup
3. Try `autogen_multi_agent_collaboration.py` to see collaboration
4. Experiment with `test_collaboration_suite.py` to validate
5. Modify examples to create your own agents

## Getting Help

- Full documentation: `README.md`
- Check setup: `python check_setup.py`
- Troubleshooting: See README.md "Troubleshooting" section
- Microsoft docs: https://learn.microsoft.com/en-us/agent-framework/

## Key Requirement

You MUST set your OpenAI API key:
```bash
export OPENAI_API_KEY='sk-...'
```

Get your key from: https://platform.openai.com/api-keys

---

**Start** Run: `python autogen_hello_agent.py`
