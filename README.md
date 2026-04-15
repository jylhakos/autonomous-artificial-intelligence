# Autonomous Artificial Intelligence

A repository for collaborative AI agents and multi-agent systems. This project provides practical examples, step-by-step tutorials, and testing methodologies for building intelligent, autonomous agents that work together to solve complex tasks.

## Get Started with Collaborative AI Agents

The best place to start is the **collaboration** folder, which contains:
- Complete step-by-step setup guide
- Working examples of multi-agent systems

```bash
cd collaboration
cat README.md  # Read the document
```

### What's Inside

This repository demonstrates how AI agents:
- Communicate and share context
- Divide complex problems into specialized tasks
- Critique and refine each other's outputs
- Work autonomously without constant human oversight

## Project Structure

```
 autonomous-artificial-intelligence/
├──  README.md                        # This file - project overview
├──  LICENSE                          # MIT License
├──  .gitignore                       # Git ignore rules
│
├──  collaboration/                   # START HERE - Complete tutorial
│   ├──  README.md                    # A step-by-step guide
│   ├──  requirements.txt             # Python dependencies
│   ├──  check_setup.py               # Environment verification tool
│   ├──  run_all_examples.sh          # Script to run all examples
│   │
│   ├── Examples:
│   ├──  autogen_hello_agent.py       # Basic single agent
│   ├──  autogen_multi_agent_collaboration.py  # Multi-agent team
│   ├──  hybrid_agent_collaboration.py         # Software dev team
│   ├──  claude_agent_basic.py        # Claude Agent SDK
│   │
│   ├── Testing:
│   ├──  test_code_collaboration.py   # Code generation test
│   ├──  test_human_feedback.py       # Human-in-loop test
│   ├──  test_collaboration_suite.py  # Complete test suite
│   │
│   └──  venv/                        # Virtual environment (create with setup)
│
├──  autonomous/                      # Additional resources
│   ├──  examples/                    # More example code
│   ├──  docs/                        # Documentation
│   └──  tests/                       # Test files
│
├──  evaluation/                      # Agent evaluation methods
├──  observability/                   # Monitoring and logging
└──  orchestration/                   # Agent orchestration patterns
```

## Key Features

### Collaborative AI Agent Examples

1. **Hello Agent** - Basic introduction to single agents
2. **Multi-Agent Collaboration** - 4 specialized agents (Planner, Researcher, Writer, Critic)
3. **Hybrid Collaboration** - Complete software development team simulation
4. **Claude Integration** - Using Claude Agent SDK with Microsoft AutoGen

### Testing Collaboration

Learn how to verify agents are truly collaborating:
- Chat log analysis and turn-taking patterns
- Task completion validation with TERMINATE signals
- Code generation and error correction
- Human-in-the-loop feedback integration
- External monitoring with AgentOps

### Complete Setup Guide

- Virtual environment setup in collaboration folder
- VS Code configuration for Python development
- Terminal integration and debugging
- API key management
- Troubleshooting common issues

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key (required)
- Anthropic API key (optional, for Claude examples)
- VS Code (recommended)

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/autonomous-artificial-intelligence.git
cd autonomous-artificial-intelligence

# 2. Navigate to collaboration folder
cd collaboration

# 3. Create virtual environment
python3 -m venv venv

# 4. Activate virtual environment
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 5. Install dependencies
pip install -U "autogen-agentchat" "autogen-ext[openai]"
pip install -r requirements.txt

# 6. Set API key
export OPENAI_API_KEY='your-api-key-here'

# 7. Verify setup
python check_setup.py

# 8. Run first example
python autogen_hello_agent.py
```

### Detailed Instructions

For setup instructions including:
- VS Code configuration
- Terminal usage
- Running in different environments
- Debugging techniques
- Testing methodologies

See: [collaboration/README.md](collaboration/README.md)

## Test

Verify everything is working:

```bash
cd collaboration
source venv/bin/activate
export OPENAI_API_KEY='your-key-here'
python autogen_hello_agent.py
```

Expected output:
```
============================================================
AutoGen Hello Agent Example
============================================================

[Task] Say 'Hello World!'

[Agent Response]
Hello World!

============================================================
Example completed successfully!
============================================================
```

## Running Examples in VS Code

### Method 1: Integrated Terminal

1. Open VS Code: `code collaboration/`
2. Select Python interpreter: `Ctrl+Shift+P` > "Python: Select Interpreter" > Choose `./venv/bin/python`
3. Open integrated terminal: `Ctrl+` `
4. Run example: `python autogen_hello_agent.py`

### Method 2: Run Button

1. Open a Python file (e.g., `autogen_hello_agent.py`)
2. Click the Run button (▶) in the top-right corner
3. Or press `Ctrl+Shift+F10`

### Method 3: Debugging

1. Set breakpoints by clicking left of line numbers
2. Press `F5` to start debugging
3. Use debugger controls to step through code

See [collaboration/README.md](collaboration/README.md) for detailed VS Code setup.

## Testing Agent Collaboration

### How to Verify Agents Are Collaborating

1. **Analyze Chat Logs** - Examine agent conversation history
2. **Check Turn-Taking** - Multiple agents should participate
3. **Validate Completion** - Look for TERMINATE signals after multiple rounds
4. **Test Code Generation** - Agents write, review, and fix code together
5. **Human Feedback** - Agents adapt based on user input

### Running Collaboration Tests

```bash
cd collaboration
source venv/bin/activate

# Run individual tests
python test_code_collaboration.py
python test_human_feedback.py

# Or run complete test suite
python test_collaboration_suite.py
```

### Example Test Output

```
======================================================================
TURN-TAKING ANALYSIS
======================================================================

Total turns: 6
Agent sequence: planner -> researcher -> writer -> critic -> writer -> planner

Turns per agent:
  planner: 2 turns
  researcher: 1 turns
  writer: 2 turns
  critic: 1 turns

 COLLABORATION DETECTED: 4 different agents participated
```

## Key Concepts

### What are Collaborative AI Agents?

Collaborative AI agents are multiple specialized AI systems that:
- **Communicate** through structured protocols
- **Share memory and context** for alignment
- **Divide complex problems** into specialized tasks
- **Critique and refine** outputs iteratively

### Why Use Multi-Agent Systems?

Instead of one monolithic agent:
- **Specialization**: Each agent excels at specific tasks
- **Efficiency**: Parallel processing of subtasks
- **Quality**: Peer review and iterative refinement
- **Scalability**: Easy to add new specialized agents

### Real-World Applications

- Software development teams
- Data analysis pipelines
- Content creation workflows
- Autonomous operations
- Research and analysis

## Framework Support

This project uses:
- **Microsoft AutoGen** (Agent Framework) - Primary framework
- **Claude Agent SDK** - Optional integration
- **OpenAI GPT Models** - Backend LLMs
- **Anthropic Claude** - Alternative LLM option

## Documentation

- [Collaboration Tutorial](collaboration/README.md) - Complete guide
- [Setup Instructions](collaboration/README.md#step-by-step-setup-guide)
- [Running Examples](collaboration/README.md#running-the-examples)
- [Testing Collaboration](collaboration/README.md#testing-agent-collaboration)
- [VS Code Setup](collaboration/README.md#vs-code-setup)
- [Troubleshooting](collaboration/README.md#troubleshooting)

## References

### Frameworks and Tools

- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) - Primary framework
- [Agent Framework Samples](https://github.com/microsoft/agent-framework/tree/main/python/samples)
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)
- [OpenAI Platform](https://platform.openai.com/)

### Documentation and Tutorials

- [Microsoft Learn: Agent Framework](https://learn.microsoft.com/en-us/agent-framework/)
- [Your First Agent Tutorial](https://learn.microsoft.com/en-us/agent-framework/get-started/your-first-agent)
- [Build AI Agents with Claude SDK](https://devblogs.microsoft.com/agent-framework/build-ai-agents-with-claude-agent-sdk-and-microsoft-agent-framework/)

### Research and Articles

- [Google's Agent-to-Agent Protocol](https://research.google/blog/advancing-agent-collaboration-with-the-agent-to-agent-a2a-protocol/)
- [Agentic AI in Operations](https://www.mckinsey.com/capabilities/operations/our-insights/the-next-frontier-of-operations-agentic-ai)
- [Multi-Agent Systems Research](https://arxiv.org/abs/2308.10848)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Support

For issues, questions, or contributions:
1. Check [collaboration/README.md](collaboration/README.md) for detailed documentation
2. Check Microsoft Agent Framework documentation

---

**Start Here**: [collaboration/README.md](collaboration/README.md) for the complete step-by-step tutorial.

## License

MIT License - see [LICENSE](LICENSE) file for details.

