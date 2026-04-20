#!/usr/bin/env python3
"""
Multi-Agent Research and Reporting System
Demonstrates message passing, context sharing, and collaboration patterns.

This example shows:
- How agents exchange structured messages
- Message tracking and logging
- Communication pattern analysis
- Multi-agent workflow coordination

Usage:
    source venv/bin/activate
    export OPENAI_API_KEY='your-key-here'
    python multi_agent_messaging_example.py
"""

import asyncio
import json
from datetime import datetime
from collections import Counter
from typing import List, Dict

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Install with: pip install -U 'autogen-agentchat' 'autogen-ext[openai]'")
    exit(1)

import os

# Check API key
if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: Please set OPENAI_API_KEY environment variable")
    print("Example: export OPENAI_API_KEY='your-key-here'")
    exit(1)


class MessageTracker:
    """Track and analyze messages exchanged between agents."""
    
    def __init__(self):
        self.messages: List[Dict] = []
    
    def log_message(self, sender: str, recipient: str, 
                    msg_type: str, content: str, metadata: Dict = None):
        """Log a message exchange between agents."""
        message = {
            "id": f"msg_{len(self.messages):03d}",
            "timestamp": datetime.now().isoformat(),
            "from": sender,
            "to": recipient,
            "type": msg_type,
            "content": content[:200],  # Truncate for display
            "metadata": metadata or {}
        }
        self.messages.append(message)
        
    def print_summary(self):
        """Print detailed communication summary with statistics."""
        print("\n" + "="*70)
        print("MESSAGE EXCHANGE SUMMARY")
        print("="*70)
        
        if not self.messages:
            print("\nNo messages tracked.")
            return
        
        # Print each message
        for msg in self.messages:
            print(f"\n[{msg['id']}] {msg['timestamp']}")
            print(f"  {msg['from']} → {msg['to']}")
            print(f"  Type: {msg['type']}")
            print(f"  Content: {msg['content']}...")
            if msg['metadata']:
                print(f"  Metadata: {msg['metadata']}")
        
        # Statistics
        print("\n" + "-"*70)
        print("COMMUNICATION STATISTICS")
        print("-"*70)
        print(f"Total messages: {len(self.messages)}")
        
        # Messages by sender
        senders = Counter(msg['from'] for msg in self.messages)
        print("\nMessages sent by agent:")
        for agent, count in senders.most_common():
            print(f"  {agent}: {count} messages")
        
        # Messages by type
        types = Counter(msg['type'] for msg in self.messages)
        print("\nMessages by type:")
        for msg_type, count in types.most_common():
            print(f"  {msg_type}: {count}")
        
        # Collaboration score
        unique_senders = len(senders)
        if unique_senders > 1:
            print(f"\n✓ COLLABORATION DETECTED: {unique_senders} agents participated")
        else:
            print(f"\n✗ NO COLLABORATION: Only 1 agent active")
    
    def export_json(self, filename: str = "message_log.json"):
        """Export messages to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.messages, f, indent=2)
        print(f"\nMessages exported to: {filename}")


async def multi_agent_research_example():
    """
    Demonstrate multi-agent communication for a research and reporting task.
    
    This example creates a team of 4 specialized agents:
    - Manager: Coordinates the workflow
    - Searcher: Conducts research
    - Writer: Creates documentation
    - Reviewer: Provides quality assurance
    """
    
    print("="*70)
    print("MULTI-AGENT RESEARCH AND REPORTING SYSTEM")
    print("="*70)
    print("\nThis example demonstrates agent communication patterns:")
    print("- Task delegation")
    print("- Information exchange")
    print("- Collaborative refinement")
    print("- Message tracking and analysis")
    
    # Initialize message tracker
    tracker = MessageTracker()
    
    # Setup OpenAI model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        # Optionally configure temperature, max_tokens, etc.
    )
    
    print("\nInitializing agents...")
    
    # Define specialized agents with clear roles
    manager = AssistantAgent(
        "manager",
        model_client=model_client,
        system_message="""You are a project manager coordinating a research team.
        
        Your responsibilities:
        - Analyze incoming requests
        - Break down tasks into subtasks
        - Delegate to appropriate team members
        - Ensure all requirements are met
        - Synthesize final deliverables
        
        When you receive a task, first acknowledge it, then create a plan 
        and delegate to the research team."""
    )
    
    searcher = AssistantAgent(
        "searcher",
        model_client=model_client,
        system_message="""You are a research specialist.
        
        Your expertise:
        - Finding and evaluating information sources
        - Identifying key concepts and definitions
        - Synthesizing research findings
        - Citing sources appropriately
        
        When given a research topic, provide comprehensive findings with 
        clear structure and references."""
    )
    
    writer = AssistantAgent(
        "writer",
        model_client=model_client,
        system_message="""You are a technical writer.
        
        Your skills:
        - Creating clear, well-structured content
        - Using appropriate formatting (markdown)
        - Organizing information logically
        - Writing for technical audiences
        
        Take research findings and transform them into polished documentation."""
    )
    
    reviewer = AssistantAgent(
        "reviewer",
        model_client=model_client,
        system_message="""You are a quality assurance reviewer.
        
        Your focus:
        - Checking accuracy and completeness
        - Ensuring clarity and readability
        - Identifying gaps or errors
        - Suggesting improvements
        
        Review content critically and provide specific, actionable feedback. 
        If content meets quality standards, approve it."""
    )
    
    # Create round-robin team
    print("Creating agent team...")
    team = RoundRobinGroupChat(
        participants=[manager, searcher, writer, reviewer],
        max_turns=12  # Allow sufficient turns for collaboration
    )
    
    # Define research task
    task = """
    Create a brief technical report (300-400 words) on Agent Communication Protocols.
    
    Requirements:
    - Cover A2A (Agent-to-Agent), MCP (Model Context Protocol), and ACP
    - Include definitions and primary use cases
    - Provide a comparison highlighting key differences
    - Use markdown formatting with headers and bullet points
    
    The report should be clear, accurate, and suitable for developers.
    """
    
    print(f"\n{'='*70}")
    print("TASK ASSIGNMENT")
    print('='*70)
    print(task)
    print("-"*70)
    
    # Log initial message
    tracker.log_message(
        sender="user",
        recipient="manager",
        msg_type="task_assignment",
        content=task,
        metadata={"priority": "normal", "deadline": "session"}
    )
    
    # Execute task
    print("\nExecuting multi-agent workflow...")
    print("(Agents will communicate to complete the task)\n")
    
    try:
        result = await team.run(task=task)
        
        # Extract conversation history for message tracking
        if hasattr(team, 'conversation_history'):
            for i, message in enumerate(team.conversation_history):
                sender = message.get('name', f'agent_{i}')
                content = message.get('content', '')
                
                # Infer message type based on position
                if i == 0:
                    msg_type = "task_decomposition"
                    recipient = "searcher"
                elif 'research' in content.lower() or 'findings' in content.lower():
                    msg_type = "research_results"
                    recipient = "writer"
                elif 'draft' in content.lower() or 'document' in content.lower():
                    msg_type = "draft_document"
                    recipient = "reviewer"
                elif 'review' in content.lower() or 'feedback' in content.lower():
                    msg_type = "review_feedback"
                    recipient = "manager"
                else:
                    msg_type = "status_update"
                    recipient = "team"
                
                tracker.log_message(
                    sender=sender,
                    recipient=recipient,
                    msg_type=msg_type,
                    content=content,
                    metadata={"turn": i+1, "conversation_id": "conv_001"}
                )
        
        # Display final result
        print("\n" + "="*70)
        print("FINAL REPORT")
        print("="*70)
        print(result)
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        return None
    
    # Print communication analysis
    tracker.print_summary()
    
    # Optional: Export to JSON
    # tracker.export_json("agent_communication_log.json")
    
    return result


def print_message_flow_diagram():
    """Print ASCII diagram showing message flow architecture."""
    diagram = """

┌─────────────────────────────────────────────────────────────────────┐
│                    MESSAGE FLOW ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────┘

USER: "Research AI agent protocols"
  │
  │ [Message Type: TASK_ASSIGNMENT]
  │ Payload: {topic, requirements, format}
  │
  ▼
┌─────────────────────────┐
│    MANAGER AGENT        │
│  - Receives task        │
│  - Analyzes scope       │
│  - Creates work plan    │
│  - Delegates to team    │
└──────────┬──────────────┘
           │
           │ [Message Type: RESEARCH_TASK]
           │ Payload: {query, sources_needed, depth}
           │
           ▼
     ┌─────────────────┐
     │  SEARCHER AGENT │
     │  - Queries info │
     │  - Finds sources│
     │  - Synthesizes  │
     └────────┬────────┘
              │
              │ [Message Type: RESEARCH_RESULTS]
              │ Payload: {findings, sources, key_points}
              │
              ▼
     ┌─────────────────┐
     │  WRITER AGENT   │
     │  - Drafts doc   │
     │  - Formats text │
     │  - Structures   │
     └────────┬────────┘
              │
              │ [Message Type: DRAFT_DOCUMENT]
              │ Payload: {document, format, metadata}
              │
              ▼
     ┌─────────────────┐
     │ REVIEWER AGENT  │
     │  - Checks quality│
     │  - Validates    │
     │  - Suggests edits│
     └────────┬────────┘
              │
              │ [Message Type: REVIEW_FEEDBACK]
              │ Payload: {status, feedback, approved}
              │
              ▼
        MANAGER AGENT
              │
              │ [Message Type: FINAL_DELIVERABLE]
              │ Payload: {result, status, summary}
              │
              ▼
            USER

┌─────────────────────────────────────────────────────────────────────┐
│                      COMMUNICATION METRICS                          │
├─────────────────────────────────────────────────────────────────────┤
│  Total Message Exchanges:  6                                        │
│  Agents Participated:      4 (Manager, Searcher, Writer, Reviewer) │
│  Conversation Turns:       ~12                                      │
│  Task Completion:          Successful                               │
│  Collaboration Pattern:    Sequential with oversight                │
└─────────────────────────────────────────────────────────────────────┘

KEY OBSERVATIONS:
- Each agent specializes in specific domain
- Messages carry structured context and metadata
- Information flows through defined pipeline
- Manager maintains oversight throughout
- Multiple refinement cycles ensure quality
    """
    print(diagram)


def print_protocol_comparison():
    """Print comparison of agent communication protocols."""
    comparison = """

┌─────────────────────────────────────────────────────────────────────┐
│           AGENT COMMUNICATION PROTOCOL COMPARISON                   │
└─────────────────────────────────────────────────────────────────────┘

╔═══════════════╦══════════════╦══════════════╦══════════════════╗
║   PROTOCOL    ║     MCP      ║     A2A      ║       ACP        ║
╠═══════════════╬══════════════╬══════════════╬══════════════════╣
║ Purpose       ║ Model-to-    ║ Agent        ║ Conversational   ║
║               ║ tool access  ║ discovery    ║ messaging        ║
║───────────────║──────────────║──────────────║──────────────────║
║ Architecture  ║ Client-      ║ Distributed  ║ Peer-to-peer or  ║
║               ║ server       ║ registry     ║ broker-based     ║
║───────────────║──────────────║──────────────║──────────────────║
║ Message       ║ JSON-RPC     ║ JSON with    ║ HTTP/JSON,       ║
║ Format        ║              ║ Agent Cards  ║ extensible MIME  ║
║───────────────║──────────────║──────────────║──────────────────║
║ Discovery     ║ Static       ║ Dynamic via  ║ Directory        ║
║               ║ config       ║ cards        ║ service          ║
║───────────────║──────────────║──────────────║──────────────────║
║ Best Use      ║ Tool & data  ║ Cross-       ║ Natural language ║
║ Case          ║ integration  ║ framework    ║ agent dialogue   ║
║               ║              ║ teams        ║                  ║
╚═══════════════╩══════════════╩══════════════╩══════════════════╝

SELECTION GUIDE:
- Use MCP when agents need standardized tool/data access
- Use A2A for multi-framework agent teams with discovery
- Use ACP for conversational, peer-to-peer agent interactions
- Protocols can be combined in hybrid architectures
    """
    print(comparison)


async def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("    AGENT COMMUNICATION AND MESSAGING DEMONSTRATION")
    print("="*70)
    
    # Run the multi-agent example
    result = await multi_agent_research_example()
    
    if result:
        # Print architecture diagrams
        print_message_flow_diagram()
        print_protocol_comparison()
        
        print("\n" + "="*70)
        print("DEMONSTRATION COMPLETE")
        print("="*70)
        print("\nKey Learnings:")
        print("✓ Agents communicate via structured messages")
        print("✓ Message tracking enables debugging and optimization")
        print("✓ Specialized agents improve task quality")
        print("✓ Protocols (MCP, A2A, ACP) enable interoperability")
        print("✓ Message flow patterns determine system architecture")
        print("\nFor more examples, see the README.md file.")
    else:
        print("\n✗ Demonstration failed. Check error messages above.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
