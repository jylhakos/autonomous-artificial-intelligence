"""
Microsoft Foundry — Short-Term Context Demo
============================================
Demonstrates how to maintain multi-turn conversational context (short-term
memory) using Foundry Agent Service Conversations without a persistent memory
store.

Within a single Conversation object, each turn's input and output are stored
as items and automatically included as context for subsequent responses.
Once the conversation is discarded, the context is gone — this is pure
short-term / in-session memory.

References:
  https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/runtime-components
  https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/memory-usage

Setup:
  1. Activate the virtual environment:
       source .venv/bin/activate        # Linux / macOS
  2. Install dependencies:
       pip install -r requirements.txt
  3. Set environment variables:
       export FOUNDRY_PROJECT_ENDPOINT="https://<resource>.services.ai.azure.com/api/projects/<project>"
       export FOUNDRY_AGENT_MODEL="gpt-4o"     # or another deployed model
  4. Run:
       python samples/foundry/foundry_short_term_context.py
"""

import os

from dotenv import load_dotenv

load_dotenv()

try:
    from azure.identity import DefaultAzureCredential
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import PromptAgentDefinition
except ImportError as exc:
    raise SystemExit(
        "azure-ai-projects is not installed.  "
        "Activate the virtual environment and run:\n  pip install -r requirements.txt"
    ) from exc

PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "")
AGENT_MODEL      = os.getenv("FOUNDRY_AGENT_MODEL", "gpt-4o")


def dry_run() -> None:
    print("=" * 60)
    print("Foundry Short-Term Context Demo — DRY RUN")
    print("=" * 60)
    print(
        "\nThis script shows multi-turn memory via Foundry Conversations.\n\n"
        "When credentials are configured, it:\n"
        "  1. Creates a prompt agent.\n"
        "  2. Creates a Conversation (durable context container).\n"
        "  3. Sends multiple turns; each builds on the previous.\n"
        "  4. Demonstrates that context is maintained within the session.\n"
        "  5. Starts a NEW conversation to show context is NOT carried over.\n\n"
        "Set FOUNDRY_PROJECT_ENDPOINT to run live.\n"
    )


def run_live(project: AIProjectClient) -> None:
    openai = project.get_openai_client()

    # Create a simple agent
    agent = project.agents.create_version(
        agent_name = "ShortTermContextAgent",
        definition = PromptAgentDefinition(
            model        = AGENT_MODEL,
            instructions = "You are a helpful assistant. Remember what the user tells you within this conversation.",
        ),
    )
    print(f"Agent created: {agent.name}")

    # ── Session A: multi-turn conversation with context ────────────────────
    print("\n[Session A] Multi-turn conversation (short-term context) …\n")
    conversation = openai.conversations.create()

    turns = [
        "My name is Alex and I love hiking.",
        "What outdoor activities would you recommend for someone like me?",
        "Do you remember my name?",
    ]

    for user_text in turns:
        print(f"  [User]  {user_text}")
        response = openai.responses.create(
            input        = user_text,
            conversation = conversation.id,
            extra_body   = {"agent_reference": {"name": agent.name, "type": "agent_reference"}},
        )
        print(f"  [Agent] {response.output_text}\n")

    # ── Session B: fresh conversation — no context carried over ───────────
    print("-" * 60)
    print("[Session B] Fresh conversation — context does NOT carry over.\n")
    new_conversation = openai.conversations.create()
    response_b = openai.responses.create(
        input        = "Do you know my name?",
        conversation = new_conversation.id,
        extra_body   = {"agent_reference": {"name": agent.name, "type": "agent_reference"}},
    )
    print(f"  [User]  Do you know my name?")
    print(f"  [Agent] {response_b.output_text}")
    print(
        "\nObservation: The agent does not know the name in Session B because\n"
        "short-term context lives only inside a single Conversation object.\n"
        "For cross-session recall, use the long-term Memory Store (foundry_memory.py).\n"
    )


if __name__ == "__main__":
    if not PROJECT_ENDPOINT:
        dry_run()
    else:
        project_client = AIProjectClient(
            endpoint   = PROJECT_ENDPOINT,
            credential = DefaultAzureCredential(),
        )
        run_live(project_client)
