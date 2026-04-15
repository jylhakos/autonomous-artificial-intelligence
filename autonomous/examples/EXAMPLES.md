# Collaborative AI Agents Examples

This document will help you set up and run the collaborative AI agent examples using Microsoft AutoGen and Claude Agent SDK.

## Prerequisites

- Python 3.8 or higher
- API Keys:
  - **OpenAI API Key** (for AutoGen examples)
  - **Anthropic API Key** (optional, for Claude examples)

## Setup Instructions

### 1. Virtual Environment Setup

The virtual environment has already been created and activated. If you need to activate it again:

```bash
cd autonomous
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies

Dependencies are already installed. If needed, reinstall with:

```bash
pip install -U "autogen-agentchat" "autogen-ext[openai]"
pip install agent-framework-claude --pre
```

### 3. Set API Keys

#### Option A: Set environment variables (Recommended)

```bash
# For AutoGen examples (Required)
export OPENAI_API_KEY='your-openai-api-key-here'

# For Claude examples (Optional)
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
```

#### Option B: Create a .env file

Create a file named `.env` in the autonomous directory:

```
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

## Available Examples

### 1. AutoGen Hello Agent (Basic)
**File:** `examples/autogen_hello_agent.py`

A simple introduction to Microsoft Agent Framework. Creates a single agent and runs a basic task.

**Run:**
```bash
python examples/autogen_hello_agent.py
```

**What it demonstrates:**
- Creating an AssistantAgent
- Configuring OpenAI model client
- Running a simple task
- Basic error handling

---

### 2. AutoGen Multi-Agent Collaboration (Advanced)
**File:** `examples/autogen_multi_agent_collaboration.py`

Demonstrates multiple specialized agents working together with different roles.

**Run:**
```bash
python examples/autogen_multi_agent_collaboration.py
```

**Agent Roles:**
- **Planner:** Task decomposition and coordination
- **Researcher:** Information gathering
- **Writer:** Content composition
- **Critic:** Quality review and feedback

**What it demonstrates:**
- Creating multiple specialized agents
- Round-robin group chat communication
- Agent collaboration and context sharing
- Iterative refinement through feedback

---

### 3. Claude Agent Basic (Claude SDK)
**File:** `examples/claude_agent_basic.py`

Shows how to use Claude Agent SDK with Microsoft Agent Framework.

**Run:**
```bash
python examples/claude_agent_basic.py
```

**Requirements:**
- ANTHROPIC_API_KEY environment variable

**What it demonstrates:**
- Basic Claude Agent with instructions
- Claude Agent with tools (Read, Write, Bash, Glob)
- Advanced reasoning capabilities
- File system operations

---

### 4. Hybrid Agent Collaboration (Most Advanced)
**File:** `examples/hybrid_agent_collaboration.py`

A comprehensive example simulating a software development team with specialized roles.

**Run:**
```bash
python examples/hybrid_agent_collaboration.py
```

**Team Structure:**
- **Product Manager:** Requirements definition
- **Tech Lead:** Architecture planning
- **Developer:** Implementation
- **QA Engineer:** Testing and validation

**What it demonstrates:**
- Complex multi-agent workflows
- Sequential task execution with context passing
- Real-world software development scenario
- Autonomous problem-solving without human intervention

---

## Running Examples

### Quick Test

Run the basic example first to verify setup:

```bash
cd /home/laptop/EXERCISES/AUTONOMOUS/autonomous-artificial-intelligence/autonomous
source venv/bin/activate
export OPENAI_API_KEY='your-key-here'
python examples/autogen_hello_agent.py
```

### Run All Examples

```bash
# Activate environment
source venv/bin/activate

# Set API keys
export OPENAI_API_KEY='your-openai-key'
export ANTHROPIC_API_KEY='your-anthropic-key'  # Optional

# Run examples one by one
python examples/autogen_hello_agent.py
python examples/autogen_multi_agent_collaboration.py
python examples/hybrid_agent_collaboration.py

# Claude example (requires ANTHROPIC_API_KEY)
python examples/claude_agent_basic.py
```

## Troubleshooting

### Issue: API Key Not Found
**Error:** `ERROR: Please set OPENAI_API_KEY environment variable`

**Solution:**
```bash
export OPENAI_API_KEY='your-key-here'
```

### Issue: Import Error
**Error:** `ModuleNotFoundError: No module named 'autogen_agentchat'`

**Solution:**
```bash
source venv/bin/activate
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

### Issue: API Rate Limits
**Error:** `Rate limit exceeded`

**Solution:**
- Wait a few moments and try again
- Check your OpenAI account has available credits
- Use a different model (e.g., gpt-4o-mini instead of gpt-4o)

### Issue: Network Connectivity
**Error:** Connection timeout or network errors

**Solution:**
- Check your internet connection
- Verify you can access OpenAI/Anthropic APIs
- Check firewall settings

## Understanding the Output

Each example provides structured output showing:

1. **Agent Creation:** Which agents are being initialized
2. **Task Execution:** What task is being processed
3. **Agent Communication:** How agents interact (in multi-agent examples)
4. **Results:** Final output from the agent(s)
5. **Summary:** Key takeaways and what was demonstrated

## Next Steps

After running these examples, you can:

1. **Modify the examples:**
   - Change agent system messages
   - Adjust the number of agents
   - Try different tasks

2. **Explore more samples:**
   - Visit: https://github.com/microsoft/agent-framework/tree/main/python/samples
   - Check: https://github.com/anthropics/claude-agent-sdk-python/tree/main/examples

3. **Build your own agents:**
   - Create custom agent roles
   - Design new workflows
   - Integrate with your applications

4. **Learn more:**
   - [Microsoft Agent Framework Documentation](https://learn.microsoft.com/en-us/agent-framework/)
   - [Your First Agent Tutorial](https://learn.microsoft.com/en-us/agent-framework/get-started/your-first-agent?pivots=programming-language-python)
   - [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python)

## Key Concepts Demonstrated

### Collaborative AI Benefits:
✓ **Specialization:** Each agent focuses on specific expertise
✓ **Efficiency:** Parallel processing and task distribution
✓ **Quality:** Iterative refinement through peer review
✓ **Scalability:** Easy to add new specialized agents

## Getting Help

- **Microsoft Agent Framework Issues:** https://github.com/microsoft/agent-framework/issues
- **Claude SDK Issues:** https://github.com/anthropics/claude-agent-sdk-python/issues
- **Documentation:** See README.md in the project root

---

