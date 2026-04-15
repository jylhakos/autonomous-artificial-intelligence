# Autonomous AI Agent Evaluation

A tutorial for evaluating autonomous AI agents using LangChain and LangSmith. This project demonstrates how to observe, evaluate, and deploy AI agents with industry best practices for monitoring and quality assurance.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Understanding Agent Evaluation](#understanding-agent-evaluation)
- [Observability with LangSmith](#observability-with-langsmith)
- [Evaluation Methodologies](#evaluation-methodologies)
- [Step-by-Step Instructions](#step-by-step-instructions)
- [Examples](#examples)
- [API Keys Setup](#api-keys-setup)
- [References](#references)

## Overview

As we move from traditional software to LLM applications and autonomous agents, each step introduces more uncertainty. Traditional software testing relies on deterministic assertions, but **agent behavior only emerges at runtime** and is captured through observability (runs, traces, and threads).

This project provides practical examples of:

- Building AI agents with **LangChain**, an open source framework with pre-built agent architecture
- Integrating **LangSmith** for observability, tracing, and evaluation
- Implementing offline evaluation (benchmarking on datasets)
- Setting up online evaluation (monitoring production)
- Using evaluation primitives: runs, traces, and threads

## Features

- Pre-configured AI agent examples with tool integration
- Automatic tracing and observability with LangSmith
- Golden dataset creation and management
- Custom evaluators for accuracy, tool usage, and efficiency
- LLM-as-judge evaluation patterns
- Production monitoring and online evaluation
- A/B testing framework for comparing agent configurations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Application                        │
│                           (main.py)                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ├─────────────────────────────────────────────┐
                     │                                             │
                     ▼                                             ▼
        ┌────────────────────────┐                   ┌────────────────────────┐
        │    Agent Creation      │                   │    Evaluation Layer    │
        │   (LangChain/Tools)    │                   │     (LangSmith)        │
        └────────┬───────────────┘                   └────────┬───────────────┘
                 │                                            │
                 ├──────────────┬──────────────┬──────────────┤
                 ▼              ▼              ▼              ▼
         ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
         │ Tools    │   │ Prompts  │   │ Traces   │   │ Datasets │
         └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

## Project Structure

```
📁 evaluation/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
├── 📄 .env.example                 # Environment variables template
├── 📄 main.py                      # Main application entry point
│
├── 📁 src/                         # Source code directory
│   ├── 📄 __init__.py              # Package initialization
│   ├── 📄 basic_agent.py           # Basic agent example
│   ├── 📄 agent_with_tracing.py    # Agent with LangSmith tracing
│   ├── 📄 agent_evaluation.py      # Offline evaluation examples
│   └── 📄 online_evaluation.py     # Online monitoring examples
│
└── 📁 venv/                        # Virtual environment (excluded from Git)
```

## Technology Stack

### Core Frameworks

- **LangChain** (>=0.3.0) - Agent framework and orchestration
- **LangSmith** (>=0.2.0) - Observability and evaluation platform
- **LangGraph** (>=0.2.0) - Graph-based agent workflows

### LLM Providers

- **Anthropic** - Claude models (primary)
- **OpenAI** - GPT models (optional)

### Supporting Libraries

- **python-dotenv** - Environment variable management
- **requests** - HTTP client
- **pydantic** - Data validation
- **langfuse** - Alternative observability (optional)


## Getting Started

### Environment Variables

```bash
# LangSmith (Required)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=agent-evaluation-demo

# LLM Provider (Required - choose one or both)
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # Optional

# Langfuse (Optional alternative)
LANGFUSE_PUBLIC_KEY=your_key_here
LANGFUSE_SECRET_KEY=your_key_here
```
### Quick Start

```bash
# Setup
./quickstart.sh

# Or manually
source venv/bin/activate
python main.py
```

### Examples

```bash
# Run specific example
python -m src.basic_agent
python -m src.agent_with_tracing
python -m src.agent_evaluation
python -m src.online_evaluation
```

### Development

```python
from src.basic_agent import create_basic_agent

# Create your own agent
agent = create_basic_agent()

# Add your tools
# Customize prompts
# Run and evaluate
```

### 1. Clone the Repository

```bash
cd evaluation
```

### 2. Create and Activate Virtual Environment

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

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit the `.env` file and add your API keys:

```bash
# LangSmith API Key (get from https://smith.langchain.com/)
LANGCHAIN_API_KEY=your_langsmith_api_key_here

# Anthropic API Key (get from https://console.anthropic.com/)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Enable tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=agent-evaluation-demo
```

### 5. Run the Application

```bash
python main.py
```

This will launch an interactive menu where you can select different examples to run.

## Understanding Agent Evaluation

### The Shift from Traditional Testing

**Traditional Software:**
- Deterministic assertions: `output == expected_output`
- Unit tests, integration tests, staging environments
- Offline tests catch most correctness issues

**Autonomous Agents:**
- Non-deterministic reasoning at runtime
- Every natural language input is unique
- Production plays a different role
- Testing reasoning, not just code paths

### What You're Testing

With agents, you're testing reasoning at different levels:

**Single-step:** Did the agent make the right decision at this moment?
**Full-turn:** Did the agent perform well in an end-to-end execution?
**Multi-turn:** Did the agent maintain context across a conversation?

## Observability with LangSmith

LangSmith is the observability layer of the LangChain ecosystem. It provides three primitives to capture non-deterministic agent reasoning:

### 1. Runs

A single execution step (one LLM call with its input/output).

**Use Case:** Capturing what the LLM did at a single step. Single-step tests often come from real production cases that error.

**Example:** Did the agent call the correct tool with the right arguments?

### 2. Traces

A complete agent execution showing all runs and their relationships.

**Use Case:** Full-turn evaluation validates complete agent executions with all their runs using the Thought-Action-Observation pattern.

**Example:** Did the agent use the correct sequence of tools to reach the answer?

### 3. Threads

Multi-turn conversations grouping multiple traces over time.

**Use Case:** Evaluating agent performance across sessions involving multiple interactions.

**Example:** Did the agent maintain context correctly across a multi-turn conversation?

### Agent Observability Powers Agent Evaluation

The traces you generate for observability are the same traces that power your evaluations:

- **Production traces become your evaluation dataset automatically**
- When a user reports a bug, traces show the exact conversation history, what the agent decided at each step, and where it went wrong
- Online evaluations run on traces you're already capturing

## Evaluation Methodologies

### Offline Evaluation

Equivalent to running unit tests before shipping. Benchmark agents on curated datasets.

**Purpose:**
- Test agent behavior before deployment
- Compare different configurations
- Validate changes don't break existing functionality

**Process:**

1. Create golden datasets from representative scenarios
2. Define evaluators to score agent outputs
3. Run experiments comparing outputs against reference data
4. Iterate and refine based on results

### Online Evaluation

Run evaluations on production data as the agent operates in real-time.

**Purpose:**
- Monitor agent behavior in production
- Catch issues like high latency or incorrect tool usage
- Track real-world performance metrics

**Process:**

1. Enable automatic tracing in production
2. Set up online evaluators in LangSmith
3. Define alert thresholds for key metrics
4. Monitor dashboards for anomalies
5. Investigate issues using trace data

## Step-by-Step Instructions

### Step 1: Enable Tracing

Set up tracing in your code to automatically log agent executions.

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "your_api_key_here"
os.environ["LANGCHAIN_PROJECT"] = "my-project"
```

This enables deep dives into tool calls and intermediate steps.

### Step 2: Create Datasets

Build a golden dataset of input-output examples representing common user scenarios and edge cases.

Use the LangSmith UI or API to:

- Collect examples from production traces
- Manually curate test cases
- Define expected outputs and behaviors

**Example:**

```python
from langsmith import Client

client = Client()

# Create dataset
dataset = client.create_dataset(
    dataset_name="agent-test-suite",
    description="Golden dataset for agent evaluation"
)

# Add examples
client.create_example(
    inputs={"input": "What's the weather in San Francisco?"},
    outputs={"expected_output": "weather information for SF"},
    dataset_id=dataset.id
)
```

### Step 3: Define Evaluators

Use built-in evaluators or define custom ones to score agent outputs.

**Common metrics:**

- **Accuracy:** Does the agent produce factually correct answers?
- **Hallucination:** Is the agent making up information?
- **Tool Usage:** Did it call the right tools with correct arguments?
- **Conciseness:** Is the response appropriately brief?
- **Relevance:** Are outputs aligned with user intent?
- **Efficiency:** Are token and latency costs optimized?

**Example Custom Evaluator:**

```python
def tool_usage_evaluator(run: Run, example: Example) -> Dict:
    expected_tools = set(example.outputs.get("expected_tools", []))
    used_tools = set(action.tool for action, _ in run.outputs["intermediate_steps"])
    
    correct = expected_tools.issubset(used_tools)
    
    return {
        "key": "tool_usage",
        "score": 1.0 if correct else 0.0,
        "comment": f"Expected: {expected_tools}, Used: {used_tools}"
    }
```

### Step 4: Run Experiments

Run the agent against the dataset, comparing output against reference data.

```python
from langsmith.evaluation import evaluate

results = evaluate(
    agent_function,
    data="agent-test-suite",
    evaluators=[accuracy_evaluator, tool_usage_evaluator],
    experiment_prefix="agent-v1",
)
```

LangSmith automatically:

- Runs your agent on each dataset example
- Applies all evaluators to the outputs
- Generates comparison reports
- Tracks experiments over time

### Step 5: Iterate and Refine

Use the LangSmith playground to tweak prompts or models based on evaluation results.

**Workflow:**

1. Review evaluation results in LangSmith dashboard
2. Identify failing test cases
3. Examine traces to understand failure modes
4. Modify prompts, tools, or model parameters
5. Re-run evaluation to verify improvements
6. Compare results across experiments

### Step 6: Deploy and Monitor

Enable online evaluation to monitor production traces.

**Setup:**

1. Ensure tracing is enabled in production
2. Configure online evaluators in LangSmith
3. Set up alerts for critical metrics
4. Monitor dashboards regularly
5. Investigate anomalies using trace data

## Examples

### Example 1: Basic Agent

Create a simple AI agent with tool-calling capabilities:

```bash
python main.py
# Select option 1
```

This demonstrates:

- Creating an agent with LangChain
- Defining custom tools
- Basic agent invocation

### Example 2: Agent with Tracing

Run an agent with full LangSmith observability:

```bash
python main.py
# Select option 2
```

This demonstrates:

- Automatic trace creation
- Capturing intermediate steps
- Viewing traces in LangSmith dashboard

### Example 3: Offline Evaluation

Evaluate agent performance using datasets:

```bash
python main.py
# Select option 3
```

This demonstrates:

- Creating golden datasets
- Defining custom evaluators
- Running evaluation experiments
- Comparing agent versions

### Example 4: Online Evaluation

Monitor agent performance in a simulated production environment:

```bash
python main.py
# Select option 4
```

This demonstrates:

- Production traffic simulation
- Real-time performance monitoring
- Alert generation
- Metrics tracking

## API Keys Setup

### Get a Free LangSmith API Key

LangSmith offers a free Developer Plan for getting started.

**Steps:**

1. Visit the [LangSmith Sign-up page](https://smith.langchain.com/)
2. Create a free account using Google, GitHub, or email
3. Click the Settings icon (gear icon) in the dashboard
4. Navigate to the API Keys section
5. Click "Create API Key"
6. Copy the generated key immediately and store it securely

**Note:** The full key is only displayed once. Store it in a password manager or your `.env` file.

### Set Environment Variables

Add to your `.env` file:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY="your_api_key_here"
export LANGCHAIN_PROJECT="my-project-name"
```

**Note:** Some documentation may use `LANGSMITH_API_KEY`, but `LANGCHAIN_API_KEY` is the standard.

### Get an Anthropic API Key

1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new API key
5. Add to your `.env` file:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## References

### LangChain

LangChain is an open source framework with pre-built agent architecture and integrations for any model.

- **LangChain Overview:** [https://www.langchain.com/langchain](https://www.langchain.com/langchain)
- **LangChain Python Docs:** [https://docs.langchain.com/oss/python/langchain/overview](https://docs.langchain.com/oss/python/langchain/overview)

### LangSmith

LangSmith is the observability and evaluation layer of the LangChain ecosystem.

- **Create Account & API Key:** [https://docs.langchain.com/langsmith/create-account-api-key](https://docs.langchain.com/langsmith/create-account-api-key)
- **Manage Organization:** [https://docs.langchain.com/langsmith/manage-organization-by-api](https://docs.langchain.com/langsmith/manage-organization-by-api)
- **Pricing Plans:** [https://www.langchain.com/pricing](https://www.langchain.com/pricing)

### Conceptual Guides

- **Agent Observability Powers Evaluation:** [https://www.langchain.com/conceptual-guides/agent-observability-powers-agent-evaluation](https://www.langchain.com/conceptual-guides/agent-observability-powers-agent-evaluation)

### Alternative Tools

- **Langfuse:** [https://langfuse.com/guides/cookbook/example_pydantic_ai_mcp_agent_evaluation](https://langfuse.com/guides/cookbook/example_pydantic_ai_mcp_agent_evaluation)
- **Langfuse GitHub:** [https://github.com/langfuse/langfuse](https://github.com/langfuse/langfuse)

## Key Takeaways

1. **Agent behavior emerges at runtime** - you can't fully test agents offline like traditional software

2. **Observability data is evaluation data** - traces generated for debugging power your evaluations

3. **Production traces become your dataset** - real user interactions provide the best test cases

4. **Test reasoning, not code** - evaluate single-step decisions, full-turn executions, and multi-turn conversations

5. **Offline + Online evaluation** - combine pre-deployment testing with production monitoring

6. **LLM-as-judge is powerful** - use capable models to evaluate agent outputs for quality, accuracy, and relevance

7. **Iterate with data** - use evaluation results to improve prompts, tools, and agent configurations

## Support

For questions or issues:

- LangChain Documentation: [https://docs.langchain.com/](https://docs.langchain.com/)
- LangSmith Support: [https://smith.langchain.com/](https://smith.langchain.com/)

