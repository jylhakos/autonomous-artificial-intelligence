# Quick Reference

## Project Structure

```
autonomous/
├── README.md                    # Main documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── .env.example                 # Environment variables template
├── setup_venv.sh                # Virtual environment setup script
├── agents/                      # Agent modules
│   └── __init__.py
├── tools/                       # Tool implementations
│   └── __init__.py
├── memory/                      # Memory systems
│   └── __init__.py
├── workflows/                   # Workflow orchestration
│   └── __init__.py
├── observability/               # Monitoring and tracing
│   └── __init__.py
├── tests/                       # Test files
├── examples/                    # Example implementations
│   ├── simple_agent.py         # Basic agent example
│   ├── multi_agent_system.py   # Multi-agent collaboration
│   └── multi_model_chat.py     # Multi-model chat UI
└── docs/                        # Documentation
    ├── setup.md                 # Setup instructions
    ├── architecture.md          # Architecture guide
    └── tutorials.md             # Step-by-step tutorials
```

## Commands

### Setup Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Using the Setup Script

```bash
# Make script executable
chmod +x setup_venv.sh

# Run setup script
./setup_venv.sh
```

### Running Examples

```bash
# Activate virtual environment first
source venv/bin/activate

# Run simple agent
python examples/simple_agent.py

# Run multi-agent system
python examples/multi_agent_system.py

# Run multi-model chat (requires API keys)
streamlit run examples/multi_model_chat.py
```

## Configuration

### Environment Variables

1. Copy the example file:

```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

### VS Code Configuration

1. Open Command Palette: `Ctrl+Shift+P` (Linux/Windows) or `Cmd+Shift+P` (macOS)
2. Type: "Python: Select Interpreter"
3. Select: `./venv/bin/python`

## Common Commands

### Development

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy agents/ tools/

# Run tests
pytest tests/

# Run specific test
pytest tests/test_agents.py -v
```

### Package Management

```bash
# Install a new package
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# List installed packages
pip list
```

### Virtual Environment

```bash
# Create
python3 -m venv venv

# Activate
source venv/bin/activate

# Deactivate
deactivate

# Remove
rm -rf venv/
```

## Key Concepts

### AI Agent Components

1. **Planning** - Breaking down goals into tasks
2. **Memory** - Short-term and long-term knowledge storage
3. **Tools** - Interfaces to perform actions
4. **Execution** - Carrying out planned tasks
5. **Feedback** - Learning from results

### Multi-Agent Collaboration Modes

**Supervisor Mode:**

- Central supervisor coordinates specialized agents
- Tasks are assigned based on agent expertise
- Results are aggregated by supervisor

**Supervisor with Routing:**

- Simple tasks go directly to relevant agent
- Complex tasks trigger multi-agent coordination
- Optimizes for efficiency

## Useful Links

### Documentation

- [Main README](../README.md)
- [Setup Guide](docs/setup.md)
- [Architecture Guide](docs/architecture.md)
- [Tutorials](docs/tutorials.md)

### External Resources

- [VS Code Agents](https://code.visualstudio.com/docs/copilot/agents/overview)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Amazon Bedrock Agents](https://aws.amazon.com/bedrock/agents/)
- [LiteLLM](https://www.litellm.ai/)

## Troubleshooting

### Virtual environment won't activate

**Solution:** Ensure you're using the correct command for your OS:

- Linux/macOS: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

### Module not found errors

**Solution:** Install dependencies:

```bash
pip install -r requirements.txt
```

### API key errors

**Solution:** Check `.env` file exists and contains valid keys

### Permission denied

**Solution:** Make script executable:

```bash
chmod +x setup_venv.sh
```

## Performance Metrics

Monitor these key metrics:

- **Response Time:** Time to complete agent tasks
- **Throughput:** Number of tasks processed per minute
- **Success Rate:** Percentage of successfully completed tasks
- **Error Rate:** Frequency of failures
- **Memory Usage:** RAM consumption during execution

## Best Practices

1. **Always use virtual environments** for isolation
2. **Keep API keys in .env file**, never commit them
3. **Test changes** before deploying
4. **Monitor agent performance** regularly
5. **Document custom agents** and tools
6. **Use type hints** for better code clarity
7. **Write tests** for critical components
8. **Log important events** for debugging

## Getting Help

1. Check the [documentation](../README.md)
2. Review [examples](../examples/)
3. Read [tutorials](docs/tutorials.md)
4. Check error logs

---

**Last Updated:** April 15, 2026
