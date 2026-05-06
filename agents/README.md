# Single-Agent, Sub-Agents and Multi-Agents

This document introduces AI agents, with practical examples built with **Google's Agent Development Kit (ADK)** to help you understand, design, and deploy your own agents, along with a tutorial on local deployment using Ollama, Open WebUI, and Docker.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Concepts Overview](#2-concepts-overview)
   - [What Is a Single Agent?](#21-what-is-a-single-agent)
   - [What Are Sub-Agents?](#22-what-are-sub-agents)
   - [What Are Multi-Agents?](#23-what-are-multi-agents)
3. [Sub-Agents In Depth](#3-sub-agents-in-depth)
   - [Orchestrator-Worker Pattern](#31-orchestrator-worker-pattern)
   - [Sub-Agent vs Agent as a Tool](#32-sub-agent-vs-agent-as-a-tool)
   - [Sub-Agent Patterns and Use Cases](#33-sub-agent-patterns-and-use-cases)
   - [What Sub-Agents Actually Do](#34-what-sub-agents-actually-do)
   - [Cross-Platform Sub-Agent Support](#35-cross-platform-sub-agent-support)
4. [Conclusion: Choosing the Right Architecture](#4-conclusion-choosing-the-right-architecture)
5. [Project Structure](#5-project-structure)
6. [Local Setup (VS Code / Linux)](#6-local-setup-vs-code--linux)
   - [Prerequisites](#61-prerequisites)
   - [Python Virtual Environment](#62-python-virtual-environment)
   - [Install Google ADK](#63-install-google-adk)
   - [API Key Configuration](#64-api-key-configuration)
   - [Run with ADK Web UI](#65-run-with-adk-web-ui)
   - [Ollama and Open WebUI (Local LLM)](#66-ollama-and-open-webui-local-llm)
   - [Docker Setup](#67-docker-setup)
7. [Prompt Engineering for Agents](#7-prompt-engineering-for-agents)
8. [Running the Sample Scripts](#8-running-the-sample-scripts)
   - [Single Agent](#81-single-agent)
   - [Sub-Agents](#82-sub-agents)
   - [Multi-Agents](#83-multi-agents)
9. [VS Code Integration](#9-vs-code-integration)
10. [References](#10-references)

---

## 1. Introduction

AI agents are programs that perceive their environment, reason about a goal, decide what action to take, and execute that action — often in a loop until the goal is met. As agent systems grow in complexity, three architectural patterns emerge:

| Pattern | Core idea | Complexity |
|---|---|---|
| **Single-Agent** | One LLM handles all reasoning and tool calls | Low |
| **Sub-Agents** | Specialized agents are directed by an orchestrator that shares session state | Medium |
| **Multi-Agents** | Multiple independent agents collaborate, each owning a domain | High |

Choosing the wrong pattern leads to unnecessary latency, cost, and maintenance overhead. This repository provides working examples of each, alongside a decision framework drawn from Microsoft's Cloud Adoption Framework and Google ADK documentation.

---

## 2. Concepts Overview

### 2.1 What Is a Single Agent?

A single-agent architecture consolidates all logic — tool calls, reasoning, and output — into one `LlmAgent`. It is the simplest starting point and should be the **default choice** until its limitations are clearly demonstrated.

**Strengths:**
- Fastest time to prototype and prove value
- Minimal orchestration overhead
- Straightforward to debug (single trace, single prompt)
- Lower token cost (no inter-agent context duplication)
- Easiest to apply governance, approval gates, and logging

**Limitations:**
- Context window limits how much data it can handle at once
- A single agent needs permissions for every action (broad privilege surface)
- Complex domains can overwhelm accuracy as context length grows

**Typical use cases:** FAQ bots, scheduled batch summarisation, fixed API sequence execution, report generation from a known data source.

```python
from google.adk.agents import LlmAgent

researcher = LlmAgent(
    name="Researcher",
    model="gemini-2.0-flash-exp",
    instruction="Research the latest trends in renewable energy.",
)
```

---

### 2.2 What Are Sub-Agents?

A **sub-agent** is a specialized `LlmAgent` that operates under the direction of a parent orchestrator agent. Sub-agents are the workers in the orchestrator-worker pattern. They receive a bounded task (and optionally the parent's shared session state) and execute it autonomously before returning a result.

Sub-agents are defined by three key properties:

- **Hierarchical delegation** — a parent agent explicitly assigns the sub-problem.
- **Shared context** — sub-agents operate within the same session as their parent and can access conversation history and shared state.
- **Stateful execution** — they are suitable for multi-step processes that require chaining of internal actions.

> A sub-agent is a delegated team member that handles a complex, multi-step process within the broader context of the parent agent's mission.
>
> — Google Cloud, [ADK architecture: When to use sub-agents versus agents as tools](https://cloud.google.com/blog/topics/developers-practitioners/where-to-use-sub-agents-versus-agents-as-tools/)

ADK workflow agents that orchestrate sub-agents:

| Orchestrator | Behaviour |
|---|---|
| `SequentialAgent` | Runs sub-agents one after another; output of each feeds into the next |
| `ParallelAgent` | Runs all sub-agents concurrently (fan-out); ideal for independent tasks |
| `LoopAgent` | Repeatedly runs sub-agents until a condition is met or max iterations reached |

---

### 2.3 What Are Multi-Agents?

A multi-agent system (MAS) deploys two or more agents for distinct tasks within a single business process. Unlike the simpler sub-agent model, a full MAS may contain multiple orchestrators, independent memory stores, different model sizes per domain, and cross-boundary communication protocols.

**Core properties of a MAS (per Google ADK):**
- **Decentralized control** — no single entity controls everything; each agent decides based on local information.
- **Local views** — agents perceive and react to their immediate context, not the entire system state.
- **Emergent behaviour** — complex, intelligent global behaviours arise from simple local interactions.

**Strengths:**
- Enables parallel development by separate teams
- Each agent can be independently scaled, upgraded, or replaced
- Enforces least-privilege security (each agent has only the access it needs)
- Supports hard compliance boundaries (e.g., a validation agent that cannot be skipped)

**Limitations:**
- Each agent handoff adds latency and a potential failure point
- Requires explicit state management, error handling, and protocol design
- Token costs multiply through redundant context and inter-agent communication
- Debugging requires distributed tracing across multiple agents

---

## 3. Sub-Agents In Depth

### 3.1 Orchestrator-Worker Pattern

The most common sub-agent architecture is the **orchestrator-worker pattern**:

```
User Request
     │
     ▼
┌──────────────────────┐
│   Orchestrator Agent │  ← Strong reasoning; breaks down the goal
│   (Root / Supervisor)│
└──────┬──────┬────────┘
       │      │
       ▼      ▼
 ┌──────────┐  ┌───────────┐
 │Sub-agent │  │Sub-agent  │  ← Execution focus; optimised for a narrow task
 │(Worker A)│  │(Worker B) │
 └──────────┘  └───────────┘
```

The orchestrator:
1. Receives a complex user goal.
2. Decomposes it into sub-tasks.
3. Dispatches each sub-task to the most suitable worker.
4. Synthesises results and either returns a final answer or initiates further sub-tasks.

Worker sub-agents are optimised for **execution, not reasoning**. They are often smaller, faster models tuned for a specific step — which improves both latency and cost.

---

### 3.2 Sub-Agent vs Agent as a Tool

Both patterns break down complex problems, but they serve different purposes based on how they handle **control and context**.

| Criterion | Agent as a Tool | Sub-Agent |
|---|---|---|
| Task complexity | Low to medium | High |
| Context and state | Isolated / stateless | Shared session state |
| Reusability | High — generic, cross-agent | Low to medium — role in a specific workflow |
| Autonomy | Minimal (request → response) | High (delegates an entire sub-problem) |
| Coupling | Loosely coupled | Tightly coupled to parent workflow |

**When to use an agent as a tool:**
- The task is a single, deterministic function (e.g., NL-to-SQL conversion, unit conversion).
- The internal logic is reusable across multiple parent agents.
- No conversational context from the parent is needed.

**When to use a sub-agent:**
- The task requires a chain of reasoning or several interaction steps.
- The agent must maintain state across those steps (e.g., iterating over search results).
- The agent needs awareness of the parent's conversation history.

> Use tools for discrete, stateless, and reusable capabilities.
> Use sub-agents to manage complex, stateful, and context-dependent processes.
>
> — Google Cloud, [ADK architecture: When to use sub-agents versus agents as tools](https://cloud.google.com/blog/topics/developers-practitioners/where-to-use-sub-agents-versus-agents-as-tools/)

---

### 3.3 Sub-Agent Patterns and Use Cases

#### Sequential (Pipeline / Linear Handoff)

A main agent breaks down a query and passes context sequentially. The output of each sub-agent becomes the input of the next.

```python
from google.adk.agents import LlmAgent, SequentialAgent

researcher = LlmAgent(name="Researcher", instruction="Find facts about the topic.")
writer = LlmAgent(name="Writer", instruction="Write a summary based on research.")
editor = LlmAgent(name="Editor", instruction="Edit the summary for clarity.")

orchestrator = SequentialAgent(
    name="BlogManager",
    sub_agents=[researcher, writer, editor],
)
```

Best for: multi-step pipelines where order matters — fetch data → analyse → summarise → review.

#### Parallel (Fan-Out)

Sub-agents run concurrently. The orchestrator aggregates results once all complete.

```python
from google.adk.agents import LlmAgent, ParallelAgent

flight_finder = LlmAgent(name="Flights", instruction="Find flights to NYC.")
hotel_finder = LlmAgent(name="Hotels", instruction="Find hotels in NYC.")

travel_planner = ParallelAgent(
    name="TravelPlanner",
    sub_agents=[flight_finder, hotel_finder],
)
```

Best for: independent tasks where latency reduction matters — calling multiple APIs, gathering data from different domains simultaneously.

#### Supervisor / Centralized Delegation

An orchestrator `LlmAgent` uses its own reasoning to dynamically decide which sub-agent handles each step. Sub-agents can also act as tools registered on the supervisor.

Best for: dynamic routing where the number of possible paths is large or not fully predetermined.

#### Loop / Iterative Refinement

A `LoopAgent` re-runs sub-agents until a condition is satisfied (e.g., a reviewer sub-agent approves the output).

Best for: iterative improvement cycles — generate → evaluate → regenerate if needed.

---

### 3.4 What Sub-Agents Actually Do

Sub-agents are optimised for **execution tasks**, not high-level reasoning:

- Extract structured data from unstructured text
- Classify inputs and route them accordingly
- Generate short-form content: emails, summaries, labels, code snippets
- Execute tool calls: web searches, database lookups, API requests
- Validate and format outputs before passing them along the pipeline
- Perform bounded Q&A or decision steps within a workflow

Because their scope is narrow, sub-agents can run on **smaller, faster models** (e.g., Gemini 2.0 Flash) without sacrificing quality. This matters significantly in pipelines with many steps:

> When you build a multi-agent workflow that autonomously plans, delegates, executes, checks, and iterates, you make 20 or even 100+ LLM calls per task. Sub-agents built for speed keep pipelines feeling responsive.

Each agent handoff in a sequential pipeline adds latency. Parallel sub-agents and lightweight models are the primary tools for managing this.

---

### 3.5 Cross-Platform Sub-Agent Support

Sub-agents are a universal pattern supported across major AI frameworks and cloud providers:

#### Google ADK (recommended for local + GCP)

```python
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent

root_agent = SequentialAgent(
    name="ResearchAndWriteWorkflow",
    sub_agents=[research_agent, writer_agent],
)
```

Deploy locally with `adk web`, or to Google Cloud Run for production scale.
See: [google/adk-samples](https://github.com/google/adk-samples/tree/main/python)

#### LangGraph (LangChain) — high control, cyclic workflows

Best when you need fine-grained state management or cyclic (non-linear) agent graphs.
See: [LangGraph Workflows and Agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)

#### CrewAI — role-playing agents with a manager pattern

Define agents by role. A `Process.hierarchical` setting automatically creates a manager sub-agent to delegate tasks to crew members.
See: [CrewAI Customize Agents](https://docs.crewai.com/en/learn/customizing-agents)

#### Microsoft Agent Framework (AutoGen successor)

Event-driven, production-ready multi-agent system for Azure deployments. Integrates with Azure Container Apps, AKS, and Azure OpenAI.
See: [microsoft/agent-framework](https://github.com/microsoft/agent-framework)

#### Amazon Bedrock (Strands SDK)

Use Bedrock AgentCore for a managed runtime with built-in memory and observability. The multi-agent collaboration feature routes tasks from a supervisor to sub-agents natively.
See: [Strands Agents SDK](https://github.com/strands-agents/sdk-python) · [AWS Deep Dive](https://aws.amazon.com/blogs/machine-learning/strands-agents-sdk-a-technical-deep-dive-into-agent-architectures-and-observability/)

#### NVIDIA NeMo — open-source LLM framework

Supports prompt learning (p-tuning), LoRA fine-tuning, and SteerLM for domain-specific sub-agents. Suitable for self-hosted inference with guardrails.
See: [NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/products/nemo/)

---

## 4. Conclusion: Choosing the Right Architecture

Microsoft's Cloud Adoption Framework provides a clear decision framework ([source](https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/ai-agents/single-agent-multiple-agents)):

> Start with a single agent to prove value quickly, then move to multi-agent structures only when the single agent hits clear limits in complexity or quality.

| Decision | Condition | Action |
|---|---|---|
| **Single agent** | Narrow domain, unified context required, speed or cost priority | Deploy single agent with tools; iterate on prompt and retrieval |
| **Prototype first** | Architecture unclear; unsure about context handling or role separation | Run controlled A/B test comparing single vs multi-agent on real workloads |
| **Multi-agent** | Hard security/compliance boundary; multiple teams own separate domains; guaranteed multi-domain scaling needed | Design isolated agents with scoped access and explicit interface contracts |

### When to choose a single agent

- The problem domain is well-defined and bounded.
- You need the fastest time to market.
- Cost constraints are significant (single agents use fewer tokens).
- Role separation (planner, reviewer, executor) can be handled by prompt switching in one agent.
- You have not yet proven that a single agent cannot handle the load.

### When to introduce sub-agents

- A task within your workflow requires a multi-step internal chain that is complex enough to warrant its own instruction context.
- The sub-task needs to maintain state across several turns.
- You want to use a smaller, faster model for a narrow execution step while keeping a larger model for orchestration.
- You are building a pipeline (sequential) or fan-out (parallel) workflow.

### When to build a full multi-agent system

1. **Security or compliance boundary** — e.g., transaction preparation and transaction validation must run in isolated environments.
2. **Multiple teams** — separate teams own separate domains and need independent deployment cycles.
3. **Planned future growth** — the solution roadmap spans more than three to five distinct functions; modularity prevents later refactoring.

---

## 5. Project Structure

```
agents/
 ▸ .venv/                        Virtual environment (excluded from git)
 ▸ single_agent/
   ◈ agent.py                    Single LlmAgent — researcher example
 ▸ sub_agents/
   ◈ agent.py                    SequentialAgent + ParallelAgent examples
 ▸ multi_agents/
   ◈ agent.py                    Supervisor pattern with 4 specialist agents
 ▪ .env.example                  Template for API key configuration
 ▪ .gitignore                    Excludes .venv/, __pycache__/, .env, etc.
 ▪ requirements.txt              Python dependencies
 ▪ README.md                     This file
```

---

## 6. Local Setup (VS Code / Linux)

### 6.1 Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.12 recommended |
| pip | Latest | Included with Python |
| Git | Any | For cloning ADK samples |
| VS Code | Latest | With Python extension |
| (Optional) Docker | 20.10+ | For containerised deployment |
| (Optional) Ollama | Latest | For local open-source LLM inference |

---

### 6.2 Python Virtual Environment

Create and activate an isolated environment to avoid dependency conflicts:

```bash
# 1. Create the virtual environment
python3 -m venv .venv

# 2. Activate it (Linux / macOS)
source .venv/bin/activate

# 3. Verify activation — the prompt will show (.venv)
python --version
```

To deactivate when finished:

```bash
deactivate
```

In **VS Code**: open the Command Palette (`Ctrl+Shift+P`) → `Python: Select Interpreter` → choose `.venv/bin/python`. The integrated terminal will activate the environment automatically when you open a new terminal.

---

### 6.3 Install Google ADK

With the virtual environment active:

```bash
pip install google-adk python-dotenv
```

Or install from this repository's requirements file:

```bash
pip install -r requirements.txt
```

Verify:

```bash
adk --version
```

Official package: [pypi.org/project/google-adk](https://pypi.org/project/google-adk/)

---

### 6.4 API Key Configuration

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set your Google AI Studio API key:

   ```bash
   GOOGLE_API_KEY=your_api_key_here
   ```

   Obtain a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

3. The sample scripts load this automatically via `python-dotenv`:

   ```python
   from dotenv import load_dotenv
   load_dotenv()  # reads .env in the current working directory
   ```

The `.env` file is listed in `.gitignore` and will never be committed to version control.

---

### 6.5 Run with ADK Web UI

The ADK ships with a browser-based development UI for inspecting agent traces, tool calls, and session state:

```bash
# From the directory containing your agent.py
cd single_agent/
adk web
```

Open the URL printed to the terminal (default: `http://localhost:8000`) in your browser. Use the **Events** or **Trace** tab to visualise how agents communicate, how context is passed between sub-agents, and where latency is introduced.

To run a single agent directly via the CLI:

```bash
adk run single_agent/
```

---

### 6.6 Ollama and Open WebUI (Local LLM)

To run agents without a cloud API key, use **Ollama** for local inference and **Open WebUI** as the chat interface.

#### Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull an open-source model (examples):

```bash
ollama pull llama3.2          # Meta LLaMA 3.2 (3B or 8B)
ollama pull mistral           # Mistral 7B
ollama pull gemma3            # Google Gemma 3 (2B or 9B)
ollama pull phi4-mini         # Microsoft Phi-4 Mini
```

Ollama exposes an OpenAI-compatible API at `http://localhost:11434`.

#### Install Open WebUI

```bash
docker run -d \
  --name open-webui \
  -p 3000:8080 \
  -v open-webui:/app/backend/data \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main
```

Access the UI at `http://localhost:3000`.

#### Use a local Ollama model with Google ADK

ADK supports LiteLLM-compatible endpoints. Point the model string to your Ollama server:

```python
from google.adk.agents import LlmAgent

researcher = LlmAgent(
    name="LocalResearcher",
    # Use the litellm prefix for Ollama models
    model="ollama/llama3.2",
    instruction="Research the topic provided and summarise key facts.",
)
```

Set the Ollama base URL in your `.env`:

```bash
OLLAMA_BASE_URL=http://localhost:11434
```

#### Essential prompts for agent interactions

When working with local models via Ollama, the instruction (system prompt) is the primary lever for controlling agent behaviour. Guidelines:

- **Be specific about role and output format.** Local models follow instructions more reliably when the expected format is explicit (e.g., "Output as a numbered list with exactly 5 items.").
- **Constrain the scope.** State what the agent should NOT do (e.g., "Do not ask clarifying questions. Produce the output immediately.").
- **Provide context anchors.** Include the session goal in the instruction so the agent does not drift across sub-agent handoffs.
- **Specify tone and length.** "Write 3 sentences for a non-technical audience." reduces variance between model runs.

Example system prompt structure for a sub-agent:

```
You are a [ROLE] specialising in [DOMAIN].
Given the [INPUT DESCRIPTION] in the conversation, [TASK DESCRIPTION].
Output format: [FORMAT].
Constraints: [WHAT NOT TO DO].
```

---

### 6.7 Docker Setup

Containerise the agents for consistent behaviour across environments.

#### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the single agent script
CMD ["python", "single_agent/agent.py"]
```

#### Build and run

```bash
# Build the image
docker build -t ai-agents:latest .

# Run with the API key injected at runtime (do not bake secrets into the image)
docker run --rm \
  -e GOOGLE_API_KEY=your_key_here \
  ai-agents:latest

# Override the command to run a different use case
docker run --rm \
  -e GOOGLE_API_KEY=your_key_here \
  ai-agents:latest python sub_agents/agent.py
```

#### Docker Compose (optional, with Ollama)

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  agents:
    build: .
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama
    command: python single_agent/agent.py

volumes:
  ollama_data:
```

---

## 7. Prompt Engineering for Agents

Prompts are the primary mechanism for defining agent behaviour. In multi-agent systems, each agent has its own `instruction` (system prompt) that governs its role, output format, and constraints.

### Anatomy of an agent instruction

```
[ROLE DEFINITION]
You are a [job title] specialising in [domain].

[TASK]
Given [what the agent will receive], [what the agent must do].

[OUTPUT FORMAT]
Output [format description, e.g., "a numbered list of 5 items", "JSON with keys X and Y"].

[CONSTRAINTS]
Do not [prohibited behaviours].
```

### Prompts for each agent type

#### Single agent

```python
instruction = """
You are a concise research assistant.
When given a topic, provide a short factual summary (3-5 sentences)
covering the key concepts, current trends, and practical applications.
Do not ask clarifying questions.
"""
```

#### Sub-agent — worker (narrow scope)

```python
instruction = """
You are a data extraction specialist.
You will receive unstructured text about a product.
Extract: product name, price, availability, and release date.
Output strictly as JSON: {"name": "", "price": "", "available": bool, "release_date": ""}.
If a field is missing, use null.
"""
```

#### Orchestrator / supervisor (broad reasoning)

```python
instruction = """
You are a project supervisor managing a team of specialist agents.
Your job is to break a complex user request into steps and delegate each step
to the appropriate specialist. Do not perform the specialist work yourself.
After all steps are complete, synthesise the results into a final answer.
Always validate quality before returning the final output.
"""
```

#### Reviewer / judge agent

```python
instruction = """
You are a critical quality reviewer.
Evaluate the content provided against: accuracy, clarity, completeness, professional tone.
If all criteria are met, output: APPROVED: <content>.
If changes are needed, output: REVISION NEEDED: <specific, actionable feedback>.
"""
```

### Prompt design principles for pipelines

- Each agent's instruction should reference what it will receive (from the shared session state) so it knows its inputs.
- Keep each instruction to a single responsibility. Compound instructions reduce reliability.
- For local models (Ollama), add explicit formatting constraints — open-source models are less instruction-following by default.
- Use the `description` field on each agent; ADK's LLM-driven delegation uses it to route requests dynamically.

---

## 8. Running the Sample Scripts

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

### 8.1 Single Agent

```bash
python single_agent/agent.py
```

What it does: sends a research prompt to a single `LlmAgent` and prints the response. No orchestration, no sub-agents. Demonstrates the baseline pattern.

Customise the prompt in `agent.py`:

```python
topic = "quantum computing applications in drug discovery"
asyncio.run(run_agent(f"Research the latest trends in {topic}."))
```

### 8.2 Sub-Agents

```bash
python sub_agents/agent.py
```

What it does:
- **Demo 1** — `SequentialAgent` runs Researcher → Writer → Editor in order. Each agent's output becomes the next agent's input via shared session state.
- **Demo 2** — `ParallelAgent` runs FlightFinder and HotelFinder concurrently. Both results are available once both complete.

### 8.3 Multi-Agents

```bash
python multi_agents/agent.py
```

What it does: a `SupervisorAgent` (implemented as a `SequentialAgent` for predictability) coordinates four specialists — ResearchAgent → AnalysisAgent → WriterAgent → ReviewerAgent. The ReviewerAgent acts as a quality gate: if the output does not meet criteria, the feedback is captured in session state for review.

---

## 9. VS Code Integration

### Recommended extensions

| Extension | Purpose |
|---|---|
| **Python** (Microsoft) | Virtual environment management, IntelliSense, debugging |
| **Pylance** | Type checking and import resolution |
| **Python Debugger** | Step-through debugging of agent execution |
| **Ruff** | Fast linting and formatting |
| **GitHub Copilot** | AI-assisted development |

### Debugging an agent in VS Code

1. Open the workspace folder in VS Code.
2. Select the `.venv` interpreter: `Ctrl+Shift+P` → `Python: Select Interpreter`.
3. Open `single_agent/agent.py`.
4. Set a breakpoint on the `runner.run_async(...)` call.
5. Press `F5` to start debugging. VS Code will stop at the breakpoint and allow you to inspect the runner, session, and event objects.

### `.vscode/launch.json` example

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Single Agent",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/single_agent/agent.py",
      "python": "${workspaceFolder}/.venv/bin/python",
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env"
    },
    {
      "name": "Sub-Agents",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/sub_agents/agent.py",
      "python": "${workspaceFolder}/.venv/bin/python",
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env"
    },
    {
      "name": "Multi-Agents",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/multi_agents/agent.py",
      "python": "${workspaceFolder}/.venv/bin/python",
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env"
    }
  ]
}
```

### ADK Web UI from VS Code terminal

```bash
# Open the VS Code integrated terminal (Ctrl+`)
cd single_agent/
adk web
# Open http://localhost:8000 in a browser
```

---

## 10. References

### Architecture and Decision Guidance

- **Single agent or multiple agents** — Microsoft Azure Cloud Adoption Framework
  https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/ai-agents/single-agent-multiple-agents

- **ADK architecture: When to use sub-agents versus agents as tools** — Google Cloud Blog
  https://cloud.google.com/blog/topics/developers-practitioners/where-to-use-sub-agents-versus-agents-as-tools/

- **Building Collaborative AI: A Developer's Guide to Multi-Agent Systems with ADK** — Google Cloud Blog
  https://cloud.google.com/blog/topics/developers-practitioners/building-collaborative-ai-a-developers-guide-to-multi-agent-systems-with-adk

- **Choosing the Right Multi-Agent Architecture** — LangChain Blog
  https://www.langchain.com/blog/choosing-the-right-multi-agent-architecture

### Google ADK

- **Google ADK Documentation**
  https://google.github.io/adk-docs/

- **Google ADK Python Samples** — GitHub
  https://github.com/google/adk-samples/tree/main/python

- **Google ADK PyPI Package**
  https://pypi.org/project/google-adk/

- **Build Agents with ADK Foundation** — Google Codelabs
  https://codelabs.developers.google.com/devsite/codelabs/build-agents-with-adk-foundation

- **Create multi-agent system with ADK, deploy in Agent Engine and get started with A2A protocol** — Google Codelabs
  https://codelabs.developers.google.com/codelabs/create-multi-agents-adk-a2a#0

- **Building a Multi-Agent System** — Google Codelabs (Production Ready AI)
  https://codelabs.developers.google.com/codelabs/production-ready-ai-roadshow/1-building-a-multi-agent-system/building-a-multi-agent-system

### Other Frameworks

- **CrewAI — Customize Agents**
  https://docs.crewai.com/en/learn/customizing-agents

- **LangGraph Workflows and Agents**
  https://docs.langchain.com/oss/python/langgraph/workflows-agents

- **Microsoft Agent Framework** — GitHub
  https://github.com/microsoft/agent-framework

- **Strands Agents SDK (Amazon)** — GitHub
  https://github.com/strands-agents/sdk-python

- **Strands Agents SDK: A Technical Deep Dive** — AWS Blog
  https://aws.amazon.com/blogs/machine-learning/strands-agents-sdk-a-technical-deep-dive-into-agent-architectures-and-observability/

- **NVIDIA NeMo**
  https://www.nvidia.com/en-us/ai-data-science/products/nemo/

### Research

- **How we built our multi-agent research system** — Anthropic Engineering
  https://www.anthropic.com/engineering/multi-agent-research-system
