# Collaboration of AI Agents

This document provides step-by-step instructions for collaborative AI agents, practical examples like Microsoft AutoGen, and testing methods to help you understand how intelligent, autonomous agents work together to solve complex tasks through communication, shared context, and distributed work.

## Table of Contents

- [Introduction](#introduction)
- [What are Collaborative AI Agents?](#what-are-collaborative-ai-agents)
- [How Collaborative AI Systems Work](#how-collaborative-ai-systems-work)
- [Agent Communication and Messaging Architecture](#agent-communication-and-messaging-architecture)
  - [Message Structure and Payload Format](#message-structure-and-payload-format)
  - [Communication Protocols for AI Agents](#communication-protocols-for-ai-agents)
    - [Model Context Protocol (MCP)](#model-context-protocol-mcp)
    - [Agent-to-Agent (A2A) Protocol](#agent-to-agent-a2a-protocol)
    - [Agent Communication Protocol (ACP)](#agent-communication-protocol-acp)
  - [Message Flow and Information Exchange Patterns](#message-flow-and-information-exchange-patterns)
  - [VS Code Agent Integration and Messaging](#vs-code-agent-integration-and-messaging)
  - [Claude Code Agent Teams](#claude-code-agent-teams)
  - [Practical Multi-Agent Communication Example](#practical-multi-agent-communication-example)
- [Key Aspects of Collaborative AI Agent Systems](#key-aspects-of-collaborative-ai-agent-systems)
- [Real-World Examples](#real-world-examples)
- [Challenges](#challenges)
- [Step-by-Step Setup Guide](#step-by-step-setup-guide)
  - [Prerequisites](#prerequisites)
  - [IMPORTANT: API Keys Required](#important-api-keys-required)
  - [Environment Setup](#environment-setup)
  - [VS Code Setup](#vs-code-setup)
- [Test](#test)
- [Running Examples in VS Code](#running-examples-in-vs-code)
- [Framework Support](#framework-support)
- [Running the Examples](#running-the-examples)
- [Testing Agent Collaboration](#testing-agent-collaboration)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [References](#references)
- [Support](#support)

## Introduction

Instead of building one monolithic agent that attempts to do everything, modern AI systems are designed as teams with clearly defined roles. Just as human teams solve complexity through collaboration, AI agents now employ the same principle to achieve faster, more accurate results through iterative, self-correcting workflows without constant human oversight.

Collaborative AI agents in autonomous systems represent a paradigm shift from single-agent approaches. By organizing into multi-agent teams—often with planners, researchers, analysts, and critics—these systems can tackle complex problems that would be difficult or impossible for a single agent to solve efficiently.

## What are Collaborative AI Agents?

Collaborative AI agents are multiple specialized AI systems designed to:

- **Communicate with each other**: Exchange information, requests, and feedback through structured protocols
- **Share memory and context**: Maintain a common understanding of tasks, goals, and progress
- **Divide complex problems into manageable subtasks**: Break down large objectives into specialized, focused work items
- **Critique, validate, and refine outputs**: Iteratively improve results through peer review and quality control

Rather than a single AI doing everything, agents with specialized roles enable more focused, efficient task execution. This approach mirrors how human organizations distribute expertise across teams.

## How Collaborative AI Systems Work

Most multi-agent systems follow a layered architecture with three core components:

### 1. Coordination Layer

A **Planner or Coordinator agent** analyzes the main objective and breaks it into subtasks. This central agent:
- Decomposes complex goals into manageable pieces
- Assigns tasks to specialized agents based on their capabilities
- Monitors progress and adjusts plans as needed
- Ensures all subtasks align with the overall objective

### 2. Shared Memory

Collaborative agents rely on shared memory systems such as vector databases. Shared memory ensures:
- **Continuity**: All agents can access historical context and previous decisions
- **Alignment**: Agents work with the same contextual understanding
- **Knowledge persistence**: Insights and learnings are preserved across tasks
- **State management**: Current progress and intermediate results are tracked

### 3. Communication Protocols

Agents exchange information using structured communication protocols, often employing:
- **Shared language and syntax**: Common vocabularies and message formats (e.g., JSON schemas)
- **Established protocols**: Standardized interaction patterns
- **Emerging tools**: Solutions like [Google's Agent-to-Agent (A2A) protocol](https://research.google/blog/advancing-agent-collaboration-with-the-agent-to-agent-a2a-protocol/) simplify connections between different frameworks

## Agent Communication and Messaging Architecture

Understanding how agents communicate is fundamental to building effective multi-agent systems. This section explores the messaging patterns, protocols, and infrastructure that enable agents to exchange information, coordinate actions, and collaborate on complex tasks.

### Message Structure and Payload Format

AI agents exchange information through structured messages that contain context, intent, and actionable data. A typical agent message includes:

**Core Message Components:**

```json
{
  "message_id": "msg_abc123",
  "timestamp": "2026-04-20T12:00:00Z",
  "sender": {
    "agent_id": "researcher_agent_01",
    "role": "researcher",
    "capabilities": ["web_search", "data_analysis"]
  },
  "recipient": {
    "agent_id": "writer_agent_01",
    "role": "writer"
  },
  "message_type": "task_result",
  "payload": {
    "task_id": "task_456",
    "status": "completed",
    "content": "Research findings on AI agent protocols...",
    "metadata": {
      "sources": ["url1", "url2"],
      "confidence": 0.92
    }
  },
  "context": {
    "conversation_id": "conv_789",
    "parent_message_id": "msg_xyz456",
    "session_id": "session_001"
  },
  "routing": {
    "priority": "normal",
    "requires_response": true,
    "timeout_ms": 30000
  }
}
```

**Message Types:**

1. **Task Definition**: Instructions for an agent to perform specific actions
2. **State Updates**: Changes in agent or system state
3. **Query/Response**: Information requests and answers
4. **Handover**: Transfer of responsibility between agents
5. **Status Notifications**: Progress updates and completion signals
6. **Error Reports**: Failure notifications and debugging information

### Communication Protocols for AI Agents

Modern multi-agent systems leverage standardized protocols to ensure interoperability across different frameworks and platforms.

#### Model Context Protocol (MCP)

**Purpose**: MCP provides a standard interface for AI models to access external tools, data sources, and services.

**Key Features:**
- Standardized context delivery to LLMs
- Secure access to external resources
- Tool and data source discovery
- Stateless communication model

**Use Case**: An agent using MCP can query databases, access APIs, or retrieve documents through a unified interface without custom integrations for each data source.

**Implementation Example:**
```python
from mcp import MCPServer, Tool

# Define a tool accessible via MCP
@Tool(name="search_knowledge_base")
async def search_kb(query: str) -> dict:
    """Search internal knowledge base."""
    results = await knowledge_base.search(query)
    return {"results": results, "count": len(results)}

# Agent uses MCP to access the tool
response = await agent.use_tool("search_knowledge_base", 
                                query="AI agent protocols")
```

**Official Resource**: [Introducing Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)

#### Agent-to-Agent (A2A) Protocol

**Purpose**: A2A defines a universal language enabling agents from different frameworks to discover, communicate, and orchestrate actions regardless of their origin.

**Key Features:**
- Cross-framework agent discovery via "Agent Cards"
- Capability advertisement and negotiation
- Secure peer-to-peer messaging
- Dynamic team formation

**Agent Card Example:**
```json
{
  "agent_id": "researcher_v1",
  "name": "Research Specialist",
  "capabilities": [
    {"name": "web_search", "version": "2.0"},
    {"name": "document_analysis", "version": "1.5"}
  ],
  "endpoints": {
    "message": "https://api.example.com/agents/researcher/message",
    "status": "https://api.example.com/agents/researcher/status"
  },
  "protocols": ["A2A/1.0", "HTTP/JSON"],
  "authentication": "Bearer"
}
```

**Communication Flow:**
```
1. Agent A publishes Agent Card to registry
2. Agent B discovers Agent A via card
3. Agent B sends task request to Agent A
4. Agent A processes and responds
5. Both agents maintain session state
```

**Official Resource**: [Google's A2A Protocol Blog](https://research.google/blog/advancing-agent-collaboration-with-the-agent-to-agent-a2a-protocol/)

#### Agent Communication Protocol (ACP)

**Purpose**: ACP facilitates peer-to-peer or broker-based interactions between agents from different frameworks using natural language-driven messages.

**Key Features:**
- RESTful API structure
- MIME-type extensibility for diverse formats
- Conversational messaging paradigm
- Framework-agnostic design

**Message Exchange Pattern:**
```http
POST /agents/writer/message HTTP/1.1
Content-Type: application/json

{
  "from": "manager_agent",
  "intent": "write_report",
  "content": {
    "topic": "AI Trends 2026",
    "format": "markdown",
    "length": "1500-2000 words",
    "sources": ["research_agent_findings.json"]
  },
  "reply_to": "https://api.example.com/agents/manager/receive"
}
```

#### Protocol Comparison

| Feature | MCP | A2A | ACP |
|---------|-----|-----|-----|
| **Focus** | Model-to-tool integration | Agent discovery & orchestration | Peer communication |
| **Architecture** | Client-server | Distributed registry | Peer-to-peer or broker |
| **Message Format** | JSON-RPC | JSON with Agent Cards | HTTP/JSON, extensible |
| **Discovery** | Static configuration | Dynamic via cards | Directory service |
| **Best For** | Tool/data access | Cross-framework teams | Conversational agents |

**Official Resource**: [Developer's Guide to AI Agent Protocols](https://developers.googleblog.com/developers-guide-to-ai-agent-protocols/)

### Message Flow and Information Exchange Patterns

Understanding how information flows between agents is critical for designing effective collaboration patterns.

#### Hierarchical Message Flow

In supervisor-based systems, messages flow through a central coordinator:

```
┌─────────────────────────────────────────────────────────────┐
│                     User / Application                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Supervisor Agent    │
              │  - Route messages    │
              │  - Aggregate results │
              │  - Manage state      │
              └──────────┬───────────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
           ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │Research  │  │ Writer   │  │ Critic   │
    │ Agent    │  │ Agent    │  │ Agent    │
    └──────────┘  └──────────┘  └──────────┘
         │             │             │
         └─────────────┴─────────────┘
                       │
                  Shared Memory
                  (Vector DB / Cache)
```

**Message Flow Steps:**
1. User sends task to Supervisor
2. Supervisor decomposes task → sends subtask to Research Agent
3. Research Agent queries tools via MCP → returns results to Supervisor
4. Supervisor forwards results to Writer Agent
5. Writer creates draft → sends to Supervisor
6. Supervisor routes draft to Critic Agent
7. Critic reviews → sends feedback through Supervisor
8. Supervisor aggregates final result → returns to user

#### Peer-to-Peer Message Flow

In decentralized systems, agents communicate directly:

```
┌─────────────────────────────────────────────────────────────┐
│                     Message Bus / Broker                    │
│              (RabbitMQ, Kafka, Redis Pub/Sub)               │
└──────┬──────────────┬──────────────┬──────────────┬─────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
  │Research│    │ Writer │    │ Critic │    │Manager │
  │ Agent  │◄──►│ Agent  │◄──►│ Agent  │◄──►│ Agent  │
  └────────┘    └────────┘    └────────┘    └────────┘
       │              │              │           │
       └──────────────┴──────────────┴───────────┘
                      │
                 Shared State
             (Distributed Cache)
```

**Message Flow Steps:**
1. Manager publishes task to message bus
2. Research & Writer agents subscribe to relevant topics
3. Research agent publishes findings
4. Writer agent consumes research → publishes draft
5. Critic agent consumes draft → publishes review
6. Writer agent refines based on feedback
7. Manager receives final output

#### Sequential Workflow Pattern

```
User Task
   │
   ▼
┌──────────────┐     Message      ┌──────────────┐
│   Agent A    │─────────────────►│   Agent B    │
│  (Analyze)   │    "Analysis     │  (Synthesize)│
└──────────────┘     Complete"    └──────────────┘
                                          │
                                  Message │
                                          ▼
                                   ┌──────────────┐
                                   │   Agent C    │
                                   │  (Validate)  │
                                   └──────────────┘
                                          │
                                          ▼
                                    Final Result
```

#### Parallel Processing Pattern

```
                    ┌──────────────┐
    ┌──────────────►│   Agent A    │─────────────┐
    │               │ (Perspective │             │
    │               │      1)      │             │
    │               └──────────────┘             │
    │                                            │
    │               ┌──────────────┐             ▼
User Task──────────►│   Agent B    │────────►Aggregator
    │               │ (Perspective │             ▲
    │               │      2)      │             │
    │               └──────────────┘             │
    │                                            │
    │               ┌──────────────┐             │
    └──────────────►│   Agent C    │─────────────┘
                    │ (Perspective │
                    │      3)      │
                    └──────────────┘
```

### VS Code Agent Integration and Messaging

VS Code leverages agents through extensions and Language Server Protocol (LSP), with GitHub Copilot being a prominent example.

#### How GitHub Copilot Communicates with Sub-Agents

**Architecture Overview:**

```
┌───────────────────────────────────────────────────────┐
│              VS Code Editor (Client)                  │
│  ┌─────────────────────────────────────────────────┐  │
│  │     GitHub Copilot Extension                    │  │
│  │  - Chat Interface                               │  │
│  │  - Code Suggestions                             │  │
│  │  - Context Gathering                            │  │
│  └─────────────────┬───────────────────────────────┘  │
└────────────────────┼──────────────────────────────────┘
                     │ JSON-RPC / WebSocket
                     ▼
┌──────────────────────────────────────────────────────┐
│          Copilot Agent Orchestrator (Server)         │
│  ┌──────────────────────────────────────────────────┐│
│  │  Primary Agent (Claude Sonnet 4.5 / GPT-4o)      ││
│  │  - Receives user messages                        ││
│  │  - Spawns specialized sub-agents                 ││
│  │  - Aggregates responses                          ││
│  └────────┬─────────────────────────────────────────┘│
│           │                                          │
│  ┌────────┼────────────────────────────────┐         │
│  │        │  Sub-Agent Pool                │         │
│  │        ▼                                │         │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │         │
│  │  │Explorer │  │ Coder   │  │Explainer│  │         │
│  │  │ Agent   │  │ Agent   │  │ Agent   │  │         │
│  │  └─────────┘  └─────────┘  └─────────┘  │         │
│  │       │            │            │       │         │
│  └───────┼────────────┼────────────┼───────┘         │
│          │            │            │                 │
│  ┌───────┴────────────┴────────────┴────────┐        │
│  │        Tool & Context Providers          │        │
│  │  - File System Access                    │        │
│  │  - Search Capabilities                   │        │
│  │  - Language Servers (Pylance, etc.)      │        │
│  └──────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

**Message Exchange Flow:**

1. **User Interaction**: User types a request in Copilot Chat interface
2. **Context Collection**: Extension gathers:
   - Open files and visible code
   - Cursor position and selected text
   - Workspace structure
   - Active errors and diagnostics
3. **Message Serialization**: Request packaged as JSON:
   ```json
   {
     "type": "chat_request",
     "message": "Refactor this function for better performance",
     "context": {
       "activeFile": "src/utils.py",
       "selection": {"start": 45, "end": 78},
       "diagnostics": [],
       "workspaceFiles": ["list", "of", "relevant", "files"]
     },
     "history": [/* previous conversation */]
   }
   ```
4. **Agent Processing**: Primary agent analyzes request and determines if sub-agents are needed
5. **Sub-Agent Invocation**: If complex, spawns specialized agents:
   ```python
   # Conceptual flow (not actual Copilot code)
   if task_needs_exploration:
       results = await explorer_agent.search_codebase(query)
   if task_needs_code_generation:
       code = await coder_agent.generate(spec, context=results)
   if task_needs_explanation:
       explanation = await explainer_agent.explain(code)
   ```
6. **Response Aggregation**: Primary agent combines sub-agent outputs
7. **Streaming Response**: Sends response back to VS Code as streaming JSON/SSE
8. **UI Rendering**: Extension displays response with syntax highlighting, code blocks, file references

**Agent Teams in VS Code (Experimental):**

Enable multi-agent teams in VS Code settings:

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

**Defining Agent Roles:**
```
User: "Spawn a researcher, a writer, and an editor agent"
```

**How This Works:**
- Primary agent creates ephemeral sub-agents with specific system prompts
- Sub-agents persist for session duration
- Communication happens through structured message passing
- Each agent maintains conversation history and context
- Handoffs include state serialization

**Agent Skills in VS Code:**

Agent Skills provide modular capabilities:
- Loaded from `.md` files with YAML frontmatter
- Available to agents via tool search
- Can be domain-specific (e.g., testing, API design, performance)
- Agent reads skill file before applying specialized knowledge

**Example Skill Definition:**
```markdown
---
name: code-review
description: Perform code reviews
applyTo: ["*.py", "*.ts"]
---

# Code Review Skill

When reviewing code, check for:
1. Correctness and logic errors
2. Performance bottlenecks
3. Security vulnerabilities
...
```

#### VS Code Agent Communication Patterns

**Pattern 1: Tool Calling**
```
User → Copilot Agent → [Tool Search] → File System Tool
                    → [Tool Execute] → Read files
                    → [Tool Result] → Generate response
```

**Pattern 2: Multi-Agent Collaboration**
```
User → Orchestrator Agent
         ├→ Search Agent (find relevant code)
         ├→ Analysis Agent (understand codebase)
         ├→ Writer Agent (generate solution)
         └→ Validator Agent (check correctness)
       → Orchestrator aggregates
       → Response to User
```

**Pattern 3: Human-in-the-Loop**
```
User → Agent → Proposes solution → User provides feedback
            ↓                            ↓
       Adjusts approach ←────────────────┘
            ↓
       Refined solution → User
```

### Claude Code Agent Teams

Claude Code (via Anthropic) supports experimental agent teams for complex workflows.

**Setup Process:**

1. **Enable Experimental Features** in VS Code `settings.json`:
   ```json
   {
     "env": {
       "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
     }
   }
   ```

2. **Define Agent Roles**: Conversationally describe the team:
   ```
   "Spawn a researcher to find information on AI protocols,
    a writer to draft documentation,
    and an editor to refine the content."
   ```

3. **Configure MCP Servers**: Provide tools for agents to use:
   ```json
   {
     "mcpServers": {
       "brave-search": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-brave-search"],
         "env": {
           "BRAVE_API_KEY": "your-key"
         }
       },
       "filesystem": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-filesystem"]
       }
     }
   }
   ```

**Agent Team Lifecycle:**

```
User Request → Claude orchestrator evaluates complexity
             ↓
      Spawns specialized agents with context
             ↓
      ┌──────┴──────┐
      │             │
   Research      Writer
   Agent         Agent
      │             │
      └──────┬──────┘
             ↓
     Results aggregated by orchestrator
             ↓
      Presented to user
             ↓
      Agents discarded (ephemeral)
```

**Communication Within Agent Teams:**

Agents communicate via structured JSON messages managed by the orchestrator:

```json
{
  "team_id": "team_xyz",
  "from_agent": "researcher",
  "to_agent": "writer",
  "message_type": "task_handover",
  "content": {
    "research_findings": "...",
    "sources": ["url1", "url2"],
    "key_points": ["point1", "point2"]
  },
  "metadata": {
    "timestamp": "2026-04-20T14:30:00Z",
    "conversation_turn": 3
  }
}
```

**Key Characteristics:**
- **Ephemeral by Design**: Agents exist only for session duration
- **Conversational Setup**: No configuration files needed
- **Context Sharing**: All agents access shared conversation memory
- **Tool Access**: Agents use MCP to access external resources
- **Transparent to User**: User sees aggregated results, not individual agent interactions

### Practical Multi-Agent Communication Example

Here's a complete example demonstrating agent communication with message tracking:

```python
"""
Multi-Agent Research and Reporting System
Demonstrates message passing, context sharing, and collaboration.
"""

import asyncio
import json
from datetime import datetime
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


class MessageTracker:
    """Track messages exchanged between agents."""
    
    def __init__(self):
        self.messages = []
    
    def log_message(self, sender: str, recipient: str, 
                    msg_type: str, content: str):
        """Log a message."""
        message = {
            "id": f"msg_{len(self.messages)}",
            "timestamp": datetime.now().isoformat(),
            "from": sender,
            "to": recipient,
            "type": msg_type,
            "content": content[:200]  # Truncate for display
        }
        self.messages.append(message)
        
    def print_summary(self):
        """Print communication summary."""
        print("\n" + "="*70)
        print("MESSAGE EXCHANGE SUMMARY")
        print("="*70)
        
        for msg in self.messages:
            print(f"\n[{msg['timestamp']}]")
            print(f"  {msg['from']} → {msg['to']}")
            print(f"  Type: {msg['type']}")
            print(f"  Content: {msg['content']}...")
        
        print(f"\nTotal messages: {len(self.messages)}")
        
        # Analyze communication patterns
        from collections import Counter
        senders = Counter(msg['from'] for msg in self.messages)
        
        print("\nMessages sent by agent:")
        for agent, count in senders.items():
            print(f"  {agent}: {count}")


async def multi_agent_research_example():
    """Demonstrate multi-agent communication for research task."""
    
    print("="*70)
    print("MULTI-AGENT RESEARCH AND REPORTING SYSTEM")
    print("="*70)
    
    # Initialize message tracker
    tracker = MessageTracker()
    
    # Setup model
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    
    # Define specialized agents
    manager = AssistantAgent(
        "manager",
        model_client=model_client,
        system_message="""You are a project manager. Break down tasks 
        and coordinate team members. Start by analyzing the request 
        and delegating to appropriate agents."""
    )
    
    searcher = AssistantAgent(
        "searcher",
        model_client=model_client,
        system_message="""You are a research specialist. When given 
        a topic, outline key information sources and findings. 
        Be specific and cite sources."""
    )
    
    writer = AssistantAgent(
        "writer",
        model_client=model_client,
        system_message="""You are a technical writer. Take research
        findings and create clear, structured content. Use markdown
        formatting."""
    )
    
    reviewer = AssistantAgent(
        "reviewer",
        model_client=model_client,
        system_message="""You are a quality assurance reviewer.
        Check content for accuracy, clarity, and completeness.
        Provide specific feedback."""
    )
    
    # Create team
    team = RoundRobinGroupChat(
        participants=[manager, searcher, writer, reviewer],
        max_turns=12
    )
    
    # Define task
    task = """
    Create a brief report on Agent Communication Protocols (A2A, MCP, ACP).
    Include definitions, use cases, and comparisons.
    """
    
    print(f"\nTask: {task}\n")
    print("-"*70)
    
    # Execute task
    result = await team.run(task=task)
    
    # Simulate message tracking from conversation history
    if hasattr(team, '_history'):
        for i, turn in enumerate(team._history):
            sender = turn.get('name', f'agent_{i}')
            content = turn.get('content', '')
            
            # Infer recipient (next in line)
            recipient = "next_agent"
            msg_type = "task_instruction" if i == 0 else "task_result"
            
            tracker.log_message(sender, recipient, msg_type, content)
    
    # Print results
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    print(result)
    
    # Print communication analysis
    tracker.print_summary()
    
    return result


# Message flow diagram (ASCII)
def print_message_flow_diagram():
    """Print ASCII diagram of message flow."""
    diagram = """
    
MESSAGE FLOW IN MULTI-AGENT SYSTEM:

User: "Research AI agent protocols"
  │
  ▼
┌─────────────────────┐
│  Manager Agent      │  Message Type: TASK_DECOMPOSITION
│  - Analyzes request │  Payload: {topic, requirements, roles}
│  - Creates plan     │
└──────────┬──────────┘
           │
           ▼ (Message: "Search for A2A, MCP, ACP protocols")
     ┌─────────────┐
     │  Searcher   │  Message Type: RESEARCH_TASK
     │  - Queries  │  Payload: {query, sources_required}
     │  - Compiles │
     └──────┬──────┘
            │
            ▼ (Message: "Findings: [research data]")
     ┌─────────────┐
     │   Writer    │  Message Type: WRITING_TASK
     │  - Drafts   │  Payload: {content, format, length}
     │  - Formats  │
     └──────┬──────┘
            │
            ▼ (Message: "Draft: [document]")
     ┌─────────────┐
     │  Reviewer   │  Message Type: REVIEW_TASK
     │  - Checks   │  Payload: {document, criteria}
     │  - Suggests │
     └──────┬──────┘
            │
            ▼ (Message: "Approved with edits")
        Manager
            │
            ▼
        User receives final report

COMMUNICATION SUMMARY:
- 5 message exchanges
- 4 agents collaborated
- 12 conversation turns
- Task completed successfully
    """
    print(diagram)


if __name__ == "__main__":
    # Run the example
    asyncio.run(multi_agent_research_example())
    
    # Print message flow diagram
    print_message_flow_diagram()
```

**Save as:** `multi_agent_messaging_example.py`

**Run:**
```bash
source venv/bin/activate
export OPENAI_API_KEY='your-key-here'
python multi_agent_messaging_example.py
```

**Expected Output Shows:**
1. Task delegation messages
2. Research findings exchange
3. Draft document handover
4. Review feedback transmission
5. Final report aggregation
6. Communication statistics and patterns

### Key Takeaways: Agent Messaging

1. **Standardized Protocols**: MCP, A2A, and ACP provide interoperability
2. **Structured Payloads**: JSON messages with context, metadata, and routing
3. **Multiple Patterns**: Hierarchical, peer-to-peer, sequential, and parallel flows
4. **VS Code Integration**: Agents communicate via JSON-RPC and tool calling
5. **Ephemeral Teams**: Modern systems spawn specialized agents dynamically
6. **Message Tracking**: Essential for debugging and optimizing collaboration
7. **Context Sharing**: Shared memory ensures alignment across agents
8. **Protocol Selection**: Choose based on architecture (centralized vs decentralized)

Effective agent communication is the foundation of robust multi-agent systems. Understanding these patterns enables you to design systems that scale, adapt, and collaborate efficiently.

## Key Aspects of Collaborative AI Agent Systems


### What are Collaborative AI Agents?

Collaborative AI agents are multiple specialized AI systems that:
- **Communicate** through structured protocols
- **Share memory and context** for alignment
- **Divide complex problems** into specialized tasks
- **Critique and refine** outputs iteratively

### What's Inside

This repository demonstrates how AI agents:
- Communicate and share context
- Divide complex problems into specialized tasks
- Critique and refine each other's outputs
- Work autonomously without constant human oversight

### Workflow Structure

A central agent (planner) breaks down large goals, assigning tasks to specialized agents (researchers, analysts, writers). Each agent focuses on its area of expertise, leading to:
- More efficient task execution
- Higher quality outputs in specialized domains
- Parallel processing of independent subtasks
- Reduced cognitive load per agent

### Communication Protocols

Agents interact via established protocols to ensure mutual understanding. Key considerations include:
- Message format standardization (JSON, XML, structured text)
- Error handling and retry mechanisms
- Asynchronous vs. synchronous communication patterns
- Protocol versioning and compatibility

### Specialization & Efficiency

Rather than a single AI doing everything, specialized agents excel at specific tasks:
- **Researcher agents**: Gather and synthesize information
- **Analyst agents**: Process data and extract insights
- **Writer agents**: Generate structured content
- **Critic agents**: Review and validate outputs
- **Executor agents**: Perform actions and operations

### Self-Correction

Agents such as critics continuously review and improve upon the work of other agents, leading to:
- Higher quality final outputs through iterative refinement
- Reduced errors and inconsistencies
- Built-in quality assurance
- Continuous learning and improvement

## Real-World Examples

### Software Development

Specialized agents independently collaborate to build and maintain software:
- **Code generation agents**: Write functions and modules based on specifications
- **Testing agents**: Create and run unit tests, integration tests
- **Debugging agents**: Detect bugs, analyze stack traces, suggest fixes
- **Documentation agents**: Generate API docs and user guides
- **Security agents**: Scan for vulnerabilities and compliance issues

### Autonomous Operations

In manufacturing and industrial settings, agents manage complex operational workflows:
- **Supply chain agents**: Handle scheduling, inventory, and logistics
- **Quality control agents**: Monitor for defects using computer vision
- **Maintenance agents**: Coordinate preventive and corrective repairs
- **Optimization agents**: Analyze production data to improve efficiency

For more details, see this [analysis of agent-driven production workflows](https://www.mckinsey.com/capabilities/operations/our-insights/the-next-frontier-of-operations-agentic-ai).

### Data Analysis

Multiple AI agents collaboratively process and analyze information:
- **Data collection agents**: Gather information from multiple sources
- **Pattern recognition agents**: Identify trends and anomalies
- **Statistical analysis agents**: Perform quantitative analysis
- **Report synthesis agents**: Generate insights
- **Visualization agents**: Create charts and dashboards

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

## Challenges

While collaborative AI agent systems offer significant benefits, several challenges must be addressed:

### Data Security

Managing privacy and data provenance across multiple agents is crucial for trust:
- Ensuring data encryption in transit and at rest
- Tracking data lineage and access patterns
- Implementing role-based access controls
- Preventing unauthorized information sharing between agents

### Regulatory Compliance

Ensuring that agent actions are transparent, auditable, and conform to legal and ethical standards:
- Maintaining audit logs of agent decisions and actions
- Implementing explainable AI techniques
- Adhering to industry-specific regulations (GDPR, HIPAA, etc.)
- Establishing governance frameworks for agent behavior

### Security Risks

The increased surface area for potential attacks on connected, autonomous systems:
- Preventing adversarial attacks on individual agents
- Securing inter-agent communication channels
- Detecting and mitigating coordinated attacks
- Implementing fail-safe mechanisms and circuit breakers

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

### Setup Guide

- Virtual environment setup in collaboration folder
- VS Code configuration for Python development
- Terminal integration and debugging
- API key management
- Troubleshooting common issues

## Step-by-Step Setup Guide

This document will walk you through setting up your environment, installing dependencies, and running collaborative AI agent examples in VS Code with Terminal.

### Prerequisites

Before you begin, ensure you have:

1. **Python 3.8 or higher** installed on your system
   ```bash
   python3 --version  # Should show 3.8 or higher
   ```

2. **VS Code** (Visual Studio Code) installed
   - Download from: https://code.visualstudio.com/

3. **API Keys**:
   - **OpenAI API Key** (Required) - Get from: https://platform.openai.com/api-keys
   - **Anthropic API Key** (Optional, for Claude examples) - Get from: https://console.anthropic.com/

4. **Git** (for cloning the repository)

### IMPORTANT: API Keys Required

**BEFORE running any examples or tests, you MUST set your API keys. The code will not work without them.**

Required:
- **OPENAI_API_KEY** - Required for all AutoGen examples

Optional:
- **ANTHROPIC_API_KEY** - Only needed for Claude Agent SDK examples

Get your API keys:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/

How to set API keys (choose one method):

**Method 1: Environment Variables (Recommended)**
```bash
# Linux/Mac
export OPENAI_API_KEY='your-openai-api-key-here'
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'  # Optional

# Windows Command Prompt
set OPENAI_API_KEY=your-openai-api-key-here
set ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Windows PowerShell
$env:OPENAI_API_KEY="your-openai-api-key-here"
$env:ANTHROPIC_API_KEY="your-anthropic-api-key-here"
```

**Method 2: Create .env file**
```bash
# Create .env file in collaboration folder
cat > .env << 'EOF'
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
EOF
```

**Verify API keys are set:**
```bash
# Linux/Mac
echo $OPENAI_API_KEY

# Windows Command Prompt
echo %OPENAI_API_KEY%

# Windows PowerShell
echo $env:OPENAI_API_KEY
```

If the command returns your key, you're ready to proceed. If it returns nothing, the key is not set.

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

## Framework Support

This project uses:
- **Microsoft AutoGen** (Agent Framework) - Primary framework
- **Claude Agent SDK** - Optional integration
- **OpenAI GPT Models** - Backend LLMs
- **Anthropic Claude** - Alternative LLM option

### Environment Setup

Follow these steps to set up your virtual environment in the collaboration folder:

#### Step 1: Navigate to the Collaboration Folder

```bash
# Clone the repository (if not already cloned)
git clone https://github.com/your-repo/autonomous-artificial-intelligence.git
cd autonomous-artificial-intelligence/collaboration

# Or if already in the project root
cd collaboration
```

#### Step 2: Create Virtual Environment

```bash
# Create a new virtual environment named 'venv'
python3 -m venv venv

# Verify it was created
ls -la venv/
```

You should see directories like `bin/`, `lib/`, `include/`, etc.

#### Step 3: Activate Virtual Environment

**On Linux/Mac:**
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

After activation, your terminal prompt should show `(venv)` prefix:
```
(venv) user@computer:~/autonomous-artificial-intelligence/collaboration$
```

#### Step 4: Upgrade pip

```bash
pip install --upgrade pip
```

#### Step 5: Install Required Packages

```bash
# Install all required packages
pip install -U "autogen-agentchat" "autogen-ext[openai]"

# Optional: Install Claude Agent SDK
pip install agent-framework-claude --pre

# Install other dependencies from requirements.txt
pip install -r requirements.txt
```

Wait for all packages to install. This may take a few minutes.

#### Step 6: Verify Installation

Run the setup checker to verify everything is installed correctly:

```bash
python check_setup.py
```

You should see output indicating which components are installed and which API keys are missing.

#### Step 7: Set API Keys

**Option A: Environment Variables (Recommended)**

**On Linux/Mac:**
```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export OPENAI_API_KEY='your-openai-api-key-here'
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'

# Or set temporarily for current session
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='sk-ant-...'
```

**On Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-openai-api-key-here
set ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

**On Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-openai-api-key-here"
$env:ANTHROPIC_API_KEY="your-anthropic-api-key-here"
```

**Option B: Create .env File**

Create a file named `.env` in the collaboration folder:

```bash
# Create .env file
cat > .env << 'EOF'
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
EOF
```

Or manually create `.env` with your favorite text editor:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

#### Step 8: Verify API Keys

Run the checker again to confirm API keys are set:

```bash
python check_setup.py
```

All checks should pass with green checkmarks.

### VS Code Setup

Configure VS Code for an optimal development experience with Python and virtual environments.

#### Step 1: Open Project in VS Code

```bash
# From the collaboration folder
code .

# Or open VS Code and use File > Open Folder > Select 'collaboration' folder
```

#### Step 2: Install VS Code Extensions

Install these recommended extensions:

1. **Python** (by Microsoft) - Essential for Python development
2. **Pylance** (by Microsoft) - Fast language server
3. **Python Environment Manager** - Easy venv switching
4. **autoDocstring** - Generate docstrings automatically

Install via VS Code:
- Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on Mac)
- Search for each extension
- Click "Install"

#### Step 3: Select Python Interpreter

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) to open Command Palette
2. Type "Python: Select Interpreter"
3. Select the interpreter from `./venv/bin/python` (or `.\venv\Scripts\python.exe` on Windows)

You should see `(venv)` in the bottom-left status bar.

#### Step 4: Configure Terminal in VS Code

1. Open integrated terminal: `Ctrl+` ` (backtick) or View > Terminal
2. The virtual environment should activate automatically
3. If not, run the activation command:
   ```bash
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

#### Step 5: Configure settings.json (Optional)

Create or edit `.vscode/settings.json` in the collaboration folder:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.testing.pytestEnabled": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "venv": false
    }
}
```

#### Step 6: Create Launch Configuration (Optional)

Create `.vscode/launch.json` for debugging:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {
                "OPENAI_API_KEY": "${env:OPENAI_API_KEY}",
                "ANTHROPIC_API_KEY": "${env:ANTHROPIC_API_KEY}"
            }
        },
        {
            "name": "Python: AutoGen Hello",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/autogen_hello_agent.py",
            "console": "integratedTerminal",
            "env": {
                "OPENAI_API_KEY": "${env:OPENAI_API_KEY}"
            }
        },
        {
            "name": "Python: Multi-Agent Collaboration",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/autogen_multi_agent_collaboration.py",
            "console": "integratedTerminal",
            "env": {
                "OPENAI_API_KEY": "${env:OPENAI_API_KEY}"
            }
        }
    ]
}
```

## Running the Examples

Now that your environment is set up, let's run the collaborative AI agent examples.

**CRITICAL: Ensure API Keys Are Set Before Running**

Before running any example or test, verify your API key is configured:

```bash
# Check if API key is set
echo $OPENAI_API_KEY  # Linux/Mac
echo %OPENAI_API_KEY%  # Windows CMD
echo $env:OPENAI_API_KEY  # Windows PowerShell

# If empty, set it now:
export OPENAI_API_KEY='your-key-here'  # Linux/Mac
```

Without a valid API key, all examples will fail with "ERROR: Please set OPENAI_API_KEY environment variable"

### Method 1: Using Terminal

#### Example 1: Hello Agent (Basic Test)

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # or appropriate command for your OS

# Verify API key is set (IMPORTANT!)
echo $OPENAI_API_KEY  # Should show your key

# Run the basic example
python autogen_hello_agent.py
```

**Expected Output:**
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

#### Example 2: Multi-Agent Collaboration

```bash
python autogen_multi_agent_collaboration.py
```

This example demonstrates 4 agents working together:
- Planner breaks down the task
- Researcher gathers information
- Writer composes content
- Critic reviews and provides feedback

Watch how agents communicate in rounds, building upon each other's contributions.

#### Example 3: Hybrid Agent Collaboration (Software Team)

```bash
python hybrid_agent_collaboration.py
```

This advanced example simulates a complete software development team:
1. Product Manager defines requirements
2. Tech Lead designs architecture
3. Developer outlines implementation
4. QA Engineer creates test plan

#### Example 4: Claude Agent (Optional)

```bash
# Requires ANTHROPIC_API_KEY
python claude_agent_basic.py
```

### Method 2: Using VS Code

#### Running in VS Code Terminal

1. Open the Python file you want to run (e.g., `autogen_hello_agent.py`)
2. Open integrated terminal (`Ctrl+` `)
3. Ensure venv is activated (you should see `(venv)` in prompt)
4. Run:
   ```bash
   python autogen_hello_agent.py
   ```

#### Running with Python Extension

1. Open the Python file
2. Click the "Run" button (▶️) in the top-right corner
3. Or press `Ctrl+Shift+F10` (Windows/Linux) or `Cmd+Shift+F10` (Mac)
4. Or right-click in the editor and select "Run Python File in Terminal"

#### Debugging in VS Code

1. Open the Python file
2. Set breakpoints by clicking left of line numbers
3. Press `F5` or click Run > Start Debugging
4. Select "Python: Current File" configuration
5. Step through code using:
   - `F10` - Step Over
   - `F11` - Step Into
   - `Shift+F11` - Step Out
   - `F5` - Continue

### Method 3: Running All Examples

Create a script to run all examples sequentially:

```bash
#!/bin/bash
# run_all_examples.sh

echo "Running all collaboration examples..."
echo ""

echo "1. Basic Hello Agent..."
python autogen_hello_agent.py
echo ""
echo "Press Enter to continue..."
read

echo "2. Multi-Agent Collaboration..."
python autogen_multi_agent_collaboration.py
echo ""
echo "Press Enter to continue..."
read

echo "3. Hybrid Agent Collaboration..."
python hybrid_agent_collaboration.py
echo ""

echo "All examples completed!"
```

Make it executable and run:
```bash
chmod +x run_all_examples.sh
./run_all_examples.sh
```

## Testing Agent Collaboration

One of the most important aspects of working with collaborative AI agents is verifying that they are truly collaborating and not just running independently. Here are methods to test and validate agent collaboration in Microsoft AutoGen.

**IMPORTANT: Set API Keys Before Running Tests**

All test scripts require valid API keys to function. Before running any tests:

```bash
# Verify API key is set
echo $OPENAI_API_KEY  # Should show your key

# If not set, configure it now
export OPENAI_API_KEY='your-key-here'  # Linux/Mac
set OPENAI_API_KEY=your-key-here  # Windows CMD
$env:OPENAI_API_KEY="your-key-here"  # Windows PowerShell
```

### 1. Analyze Chat Logs

The most direct method is examining the conversation history between agents.

#### Method: Inspect conversation history

Add this code to your agent scripts to print detailed chat logs:

```python
# After running the agents
print("\n" + "="*70)
print("CHAT LOG ANALYSIS")
print("="*70)

# For team-based collaboration
if hasattr(team, 'conversation_history'):
    for i, message in enumerate(team.conversation_history, 1):
        print(f"\nMessage {i}:")
        print(f"  From: {message.get('name', 'Unknown')}")
        print(f"  Role: {message.get('role', 'Unknown')}")
        print(f"  Content: {message.get('content', '')[:200]}...")  # First 200 chars

# For agent sessions
if hasattr(user_proxy, 'chat_messages'):
    print("\nFull conversation:")
    for agent_name, messages in user_proxy.chat_messages.items():
        print(f"\n{agent_name}'s messages:")
        for msg in messages:
            print(f"  - {msg}")
```

#### What to Look For:

**Turn-Taking:**
- Messages should alternate between different agents
- Example pattern: Planner -> Researcher -> Writer -> Critic -> Planner
- If you only see one agent, collaboration isn't happening

**Context Building:**
- Later messages should reference earlier messages
- Agents should build upon each other's ideas
- Look for phrases like "Based on the previous analysis..." or "As mentioned by..."

**Task Completion Signal:**
- Look for "TERMINATE" or similar completion signals
- Should appear after multiple rounds, not immediately

### 2. Check Turn-Taking Patterns

Create a visualization of agent interactions:

```python
def analyze_turn_taking(conversation_history):
    """Analyze and visualize turn-taking patterns."""
    print("\n" + "="*70)
    print("TURN-TAKING ANALYSIS")
    print("="*70)
    
    agents_sequence = []
    for msg in conversation_history:
        agent_name = msg.get('name', 'Unknown')
        agents_sequence.append(agent_name)
    
    print(f"\nTotal turns: {len(agents_sequence)}")
    print(f"Agent sequence: {' -> '.join(agents_sequence)}")
    
    # Count turns per agent
    from collections import Counter
    turn_counts = Counter(agents_sequence)
    
    print("\nTurns per agent:")
    for agent, count in turn_counts.items():
        print(f"  {agent}: {count} turns")
    
    # Check for collaboration
    unique_agents = len(turn_counts)
    if unique_agents > 1:
        print(f"\n✓ COLLABORATION DETECTED: {unique_agents} different agents participated")
    else:
        print(f"\n✗ NO COLLABORATION: Only 1 agent participated")
    
    return turn_counts

# Usage
analyze_turn_taking(team.conversation_history)
```

### 3. Validate Task Completion

Verify that tasks complete through collaborative iterations:

```python
def validate_collaboration_quality(conversation_history, expected_min_turns=3):
    """Validate that collaboration occurred with sufficient depth."""
    print("\n" + "="*70)
    print("COLLABORATION QUALITY VALIDATION")
    print("="*70)
    
    total_turns = len(conversation_history)
    print(f"\nTotal conversation turns: {total_turns}")
    
    if total_turns < expected_min_turns:
        print(f"✗ INSUFFICIENT COLLABORATION: Expected at least {expected_min_turns} turns")
        return False
    
    # Check for termination signal
    last_message = conversation_history[-1].get('content', '')
    has_termination = 'TERMINATE' in last_message or 'completed' in last_message.lower()
    
    if has_termination:
        print("✓ Task properly terminated after collaboration")
    else:
        print("⚠ No clear termination signal found")
    
    # Check for different agent roles
    agents_involved = set(msg.get('name') for msg in conversation_history)
    print(f"\nAgents involved: {', '.join(agents_involved)}")
    
    if len(agents_involved) >= 2:
        print(f"✓ Multiple agents collaborated ({len(agents_involved)} agents)")
        return True
    else:
        print("✗ Single agent execution (no collaboration)")
        return False

# Usage
is_valid = validate_collaboration_quality(team.conversation_history)
```

### 4. Code Generation & Correction Test

Test if agents can collaboratively write and fix code:

```python
"""
Test: Code Generation with Error Correction
This tests if agents can collaborate to write and debug code.
"""

import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def test_code_collaboration():
    """Test collaborative code generation and debugging."""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    
    # Code Writer Agent
    coder = AssistantAgent(
        "coder",
        model_client=model_client,
        system_message="""You are a Python developer. Write clean, working code.
        When you receive error messages, analyze them and fix the code."""
    )
    
    # Code Executor Agent (simulated)
    executor = AssistantAgent(
        "executor",
        model_client=model_client,
        system_message="""You are a code execution simulator. When you receive code:
        1. Analyze it for potential errors
        2. Report any issues found
        3. If correct, confirm it would execute successfully"""
    )
    
    task = """Write a Python function to calculate fibonacci numbers.
    The executor will check it for errors."""
    
    print("="*70)
    print("CODE COLLABORATION TEST")
    print("="*70)
    print(f"\nTask: {task}\n")
    
    # Create team
    from autogen_agentchat.teams import RoundRobinGroupChat
    team = RoundRobinGroupChat(participants=[coder, executor], max_turns=6)
    
    result = await team.run(task=task)
    
    # Analyze collaboration
    print("\n" + "="*70)
    print("COLLABORATION ANALYSIS")
    print("="*70)
    
    conversation = team.conversation_history if hasattr(team, 'conversation_history') else []
    
    code_written = any('def' in msg.get('content', '') for msg in conversation)
    errors_reported = any('error' in msg.get('content', '').lower() or 
                         'fix' in msg.get('content', '').lower() 
                         for msg in conversation)
    
    print(f"\nCode was written: {'✓' if code_written else '✗'}")
    print(f"Errors/fixes discussed: {'✓' if errors_reported else '✗'}")
    print(f"Total interaction rounds: {len(conversation)}")
    
    if code_written and len(conversation) >= 2:
        print("\n✓ Collaborative code development occurred")
    else:
        print("\n✗ Collaboration was insufficient")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_code_collaboration())
```

Save this as `test_code_collaboration.py` and run:
```bash
python test_code_collaboration.py
```

### 5. Human-in-the-Loop Feedback

Test interactive collaboration with human input:

```python
"""
Test: Human-in-the-Loop Collaboration
This demonstrates how human feedback forces agents to adjust approaches.
"""

import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def test_human_in_loop():
    """Test collaboration with human feedback."""
    
    print("="*70)
    print("HUMAN-IN-THE-LOOP COLLABORATION TEST")
    print("="*70)
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    
    assistant = AssistantAgent(
        "assistant",
        model_client=model_client,
        system_message="You are a helpful assistant. Respond to feedback and adjust your approach."
    )
    
    # Simulate human feedback loop
    task = "Explain quantum computing in simple terms."
    
    print(f"\nInitial Task: {task}\n")
    print("-"*70)
    
    # First attempt
    print("\n[Round 1] Initial Explanation:")
    response1 = await assistant.run(task=task)
    print(response1)
    
    # Human feedback (simulated)
    feedback = """That's too technical. Please explain it using an analogy 
    that a 10-year-old would understand."""
    
    print("\n" + "-"*70)
    print(f"\n[Human Feedback]: {feedback}\n")
    print("-"*70)
    
    # Adjusted attempt
    print("\n[Round 2] Adjusted Explanation:")
    adjusted_task = f"{task}\n\nFeedback received: {feedback}\nPlease adjust your explanation."
    response2 = await assistant.run(task=adjusted_task)
    print(response2)
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("\n✓ Agent received human feedback")
    print("✓ Agent adjusted approach based on feedback")
    print("✓ Interactive collaboration demonstrated")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_human_in_loop())
```

Save as `test_human_feedback.py` and run:
```bash
python test_human_feedback.py
```

### 6. External Monitoring Tools

For production systems, use external monitoring tools to track agent interactions.

#### AgentOps Integration

Install and configure AgentOps for monitoring:

```bash
pip install agentops
```

```python
"""
Test: AgentOps Monitoring
Monitor multi-agent interactions with external tooling.
"""

import agentops
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def test_with_monitoring():
    """Run agents with AgentOps monitoring."""
    
    # Initialize AgentOps
    agentops.init(api_key="your-agentops-api-key", default_tags=["collaboration-test"])
    
    print("="*70)
    print("COLLABORATION WITH MONITORING")
    print("="*70)
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    
    # Create agents
    planner = AssistantAgent("planner", model_client,
                            system_message="You plan tasks.")
    worker = AssistantAgent("worker", model_client,
                           system_message="You execute tasks.")
    
    # Create monitored team
    team = RoundRobinGroupChat(participants=[planner, worker], max_turns=4)
    
    # Run task
    result = await team.run(task="Create a simple website landing page")
    
    # End session
    agentops.end_session("Success")
    
    print("\n✓ Session logged to AgentOps dashboard")
    print("✓ View interaction tree and statistics at: https://app.agentops.ai")

# Run with monitoring
if __name__ == "__main__":
    asyncio.run(test_with_monitoring())
```

#### What AgentOps Provides:

- **Session-wide statistics**: Total tokens, cost, duration
- **Call tree visualization**: See agent interaction flow
- **Error tracking**: Identify failures in collaboration
- **Performance metrics**: Response times, success rates

### 7. Quick Collaboration Verification Checklist

Use this checklist to verify collaboration in your examples:

```
□ Multiple agents instantiated with different roles
□ Agents exchange at least 3-5 messages
□ Messages reference previous agent outputs
□ Different agents contribute unique information
□ Task completes with TERMINATE or completion signal
□ Chat logs show turn-taking pattern
□ Agents build upon each other's work
□ Final output incorporates multiple agent contributions
```

### 8. Create a Collaboration Test Suite

Create `test_collaboration_suite.py`:

```python
"""
Test Suite
Run all collaboration tests in sequence.
"""

import asyncio
from test_code_collaboration import test_code_collaboration
from test_human_feedback import test_human_in_loop

async def run_all_tests():
    """Run complete collaboration test suite."""
    
    print("\n" + "="*70)
    print("AUTONOMOUS AI COLLABORATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Code Collaboration Test", test_code_collaboration),
        ("Human-in-Loop Test", test_human_in_loop),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n\nRunning: {test_name}")
        print("-"*70)
        try:
            await test_func()
            results.append((test_name, "PASSED"))
        except Exception as e:
            print(f"✗ Test failed: {e}")
            results.append((test_name, "FAILED"))
    
    # Summary
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, status in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
```

Run the full suite:
```bash
python test_collaboration_suite.py
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Virtual Environment Not Activating

**Symptoms:** Prompt doesn't show `(venv)`, packages not found

**Solutions:**
```bash
# Linux/Mac - Make sure you're using source
source venv/bin/activate

# Windows - Try different methods
venv\Scripts\activate.bat  # Command Prompt
venv\Scripts\Activate.ps1  # PowerShell

# If PowerShell gives execution policy error
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Issue: API Key Not Found

**Symptoms:** `ERROR: Please set OPENAI_API_KEY environment variable`

**Solutions:**
```bash
# Verify key is set
echo $OPENAI_API_KEY  # Linux/Mac
echo %OPENAI_API_KEY%  # Windows CMD
$env:OPENAI_API_KEY  # Windows PowerShell

# If not set, export it
export OPENAI_API_KEY='your-key-here'  # Linux/Mac

# Or use .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

#### Issue: Module Not Found Errors

**Symptoms:** `ModuleNotFoundError: No module named 'autogen_agentchat'`

**Solutions:**
```bash
# Make sure venv is activated
source venv/bin/activate  # or appropriate command

# Reinstall packages
pip install -U "autogen-agentchat" "autogen-ext[openai]"

# Verify installation
pip list | grep autogen
```

#### Issue: Rate Limit Errors

**Symptoms:** `Rate limit exceeded` or 429 errors

**Solutions:**
- Wait a few minutes before retrying
- Use `gpt-4o-mini` instead of `gpt-4o` (cheaper, faster)
- Check your OpenAI usage limits
- Consider adding delays between requests

#### Issue: No Collaboration Detected

**Symptoms:** Only one agent responds, no turn-taking

**Solutions:**
- Increase `max_turns` in RoundRobinGroupChat
- Check agent system messages are different and complementary
- Ensure task requires multiple perspectives
- Add explicit instructions for agents to collaborate

#### Issue: VS Code Not Finding Python Interpreter

**Symptoms:** "Python is not installed" or wrong interpreter

**Solutions:**
1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose `./venv/bin/python` (or `.\venv\Scripts\python.exe` on Windows)
4. Reload VS Code window if needed

## Project Structure

```
📦 collaboration/
├── 📄 README.md                               # This document
├── 📄 QUICKSTART.md                           # Quick start guide
├── 📄 requirements.txt                        # Python dependencies
├── 📄 check_setup.py                          # Environment verification tool
├── 📄 run_all_examples.sh                     # Script to run all examples
├── 📄 .gitignore                              # Git ignore patterns
├── 📄 .env                                    # API keys (create this, not in git)
│
├── 📄 autogen_hello_agent.py                  # Basic single agent example
├── 📄 autogen_multi_agent_collaboration.py    # Multi-agent team example
├── 📄 hybrid_agent_collaboration.py           # Software development team simulation
├── 📄 claude_agent_basic.py                   # Claude Agent SDK examples
├── 📄 multi_agent_messaging_example.py        # ✨ Agent communication demonstration
│
├── 📄 test_code_collaboration.py              # Code generation & correction test
├── 📄 test_human_feedback.py                  # Human-in-the-loop test
├── 📄 test_collaboration_suite.py             # Complete test suite
│
└── 📁 venv/                                   # Virtual environment (excluded from git)
    ├── 📁 bin/                                # Executables (Scripts/ on Windows)
    ├── 📁 lib/                                # Installed packages
    └── 📁 include/                            # C headers
```

## References

### Documentation and Tutorials

- [Microsoft Learn: Agent Framework](https://learn.microsoft.com/en-us/agent-framework/) - Official documentation
- [Your First Agent Tutorial](https://learn.microsoft.com/en-us/agent-framework/get-started/your-first-agent?pivots=programming-language-python) - Getting started guide
- [Build AI Agents with Claude SDK and Agent Framework](https://devblogs.microsoft.com/agent-framework/build-ai-agents-with-claude-agent-sdk-and-microsoft-agent-framework/) - Integration guide

### Frameworks and Tools

- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) - A programming framework for agentic AI
- [Microsoft Agent Framework Samples](https://github.com/microsoft/agent-framework/tree/main/python/samples) - Practical examples and use cases
- [Claude Agent SDK for Python](https://github.com/anthropics/claude-agent-sdk-python) - Anthropic's agent development kit
- [Claude Agent SDK Examples](https://github.com/anthropics/claude-agent-sdk-python/tree/main/examples) - Example implementations
- [OpenAI Platform](https://platform.openai.com/)

### Research and Articles

- [Google's Agent-to-Agent (A2A) Protocol](https://research.google/blog/advancing-agent-collaboration-with-the-agent-to-agent-a2a-protocol/) - Advancing agent collaboration
- [Developer's Guide to AI Agent Protocols](https://developers.googleblog.com/developers-guide-to-ai-agent-protocols/) - An overview of MCP, A2A, and ACP
- [Introducing Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) - Anthropic's MCP announcement
- [Agentic AI in Operations](https://www.mckinsey.com/capabilities/operations/our-insights/the-next-frontier-of-operations-agentic-ai) - McKinsey analysis
- [The Rise of Multi-Agent Systems](https://arxiv.org/abs/2308.10848) - Academic perspective on collaborative agents

## Support

For issues, questions, or contributions, check the [Microsoft Agent Framework documentation](https://github.com/microsoft/agent-framework).

---

**License**: MIT