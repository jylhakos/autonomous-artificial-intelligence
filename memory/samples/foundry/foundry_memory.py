"""
Microsoft Foundry Agent Service — Memory Sample
================================================
Demonstrates how Microsoft Azure AI Foundry (formerly Azure AI Foundry) manages
agent memory across sessions using its managed Memory Store API (preview).

Concepts demonstrated:
  - MemoryStoreDefaultDefinition  : long-term memory store backed by a chat
                                    model + embedding model
  - MemorySearchPreviewTool       : attaches memory to a prompt agent so it can
                                    read/write memories during conversations
  - Conversations                 : durable objects that maintain multi-turn
                                    context across sessions
  - Scope parameter               : isolates each user's memories
  - Short-term context            : maintained via conversation items (threads)
  - Long-term persistence         : managed memory store survives across sessions

Architecture summary:
  ┌──────────────────────────────────────────────────────────────┐
  │  Microsoft Foundry Agent Service Memory Architecture         │
  │                                                              │
  │  Short-term (in-conversation):                               │
  │    Conversation object → stores message items per turn       │
  │    previous_response_id → chains responses without an object │
  │                                                              │
  │  Long-term (cross-session):                                  │
  │    MemoryStore → chat summaries + user profiles              │
  │    MemorySearchPreviewTool → agent queries memories per turn │
  └──────────────────────────────────────────────────────────────┘

References:
  https://learn.microsoft.com/en-us/azure/foundry/agents/concepts/runtime-components
  https://learn.microsoft.com/en-us/azure/foundry/agents/how-to/memory-usage
  https://devblogs.microsoft.com/foundry/azure-ai-mem0-integration/

Environment variables required:
  FOUNDRY_PROJECT_ENDPOINT                   — project endpoint URL
  MEMORY_STORE_CHAT_MODEL_DEPLOYMENT_NAME    — e.g. gpt-5.2 or gpt-4o
  MEMORY_STORE_EMBEDDING_MODEL_DEPLOYMENT_NAME — e.g. text-embedding-3-small

Setup:
  1. Activate the virtual environment:
       source .venv/bin/activate        # Linux / macOS
       .venv\\Scripts\\activate           # Windows
  2. Install dependencies:
       pip install -r requirements.txt
  3. Copy the example environment file and fill in your values:
       cp .env.example .env
  4. Run this script:
       python samples/foundry/foundry_memory.py
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------
try:
    from azure.identity import DefaultAzureCredential
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import (
        PromptAgentDefinition,
        MemoryStoreDefaultDefinition,
        MemoryStoreDefaultOptions,
        MemorySearchPreviewTool,
        MemorySearchOptions,
    )
except ImportError as exc:
    raise SystemExit(
        "azure-ai-projects is not installed.  "
        "Activate the virtual environment and run:\n"
        "  pip install -r requirements.txt"
    ) from exc

# ---------------------------------------------------------------------------
# Configuration — read from environment
# ---------------------------------------------------------------------------
PROJECT_ENDPOINT   = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "")
CHAT_MODEL         = os.getenv("MEMORY_STORE_CHAT_MODEL_DEPLOYMENT_NAME",    "gpt-4o")
EMBEDDING_MODEL    = os.getenv("MEMORY_STORE_EMBEDDING_MODEL_DEPLOYMENT_NAME", "text-embedding-3-small")
MEMORY_STORE_NAME  = "demo-memory-store"
USER_SCOPE         = "demo_user_001"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env() -> bool:
    """Return True when all required environment variables are present."""
    missing = [v for v in ("FOUNDRY_PROJECT_ENDPOINT",) if not os.getenv(v)]
    if missing:
        print(
            "WARNING: The following environment variables are not set:\n"
            + "\n".join(f"  {v}" for v in missing)
            + "\nThis script will run in DRY-RUN mode (no real API calls).\n"
            + "See the README.md for how to configure your Foundry project.\n"
        )
        return False
    return True


def _dry_run_demo() -> None:
    """Illustrative walk-through printed when credentials are missing."""
    print("=" * 60)
    print("Microsoft Foundry Memory Demo — DRY RUN")
    print("=" * 60)
    print(
        "\nStep 1: Create a Memory Store\n"
        "  memory_store = project_client.beta.memory_stores.create(\n"
        "      name='demo-memory-store',\n"
        "      definition=MemoryStoreDefaultDefinition(\n"
        "          chat_model='gpt-4o',\n"
        "          embedding_model='text-embedding-3-small',\n"
        "          options=MemoryStoreDefaultOptions(\n"
        "              chat_summary_enabled=True,\n"
        "              user_profile_enabled=True,\n"
        "          ),\n"
        "      ),\n"
        "  )\n"
        "\nStep 2: Attach the Memory Search Tool to a Prompt Agent\n"
        "  tool = MemorySearchPreviewTool(\n"
        "      memory_store_name='demo-memory-store',\n"
        "      scope='demo_user_001',\n"
        "      update_delay=60,\n"
        "  )\n"
        "  agent = project_client.agents.create_version(\n"
        "      agent_name='MemoryAgent',\n"
        "      definition=PromptAgentDefinition(\n"
        "          model='gpt-4o',\n"
        "          instructions='You are a helpful assistant with memory.',\n"
        "          tools=[tool],\n"
        "      ),\n"
        "  )\n"
        "\nStep 3: Session A — store a preference\n"
        "  conversation = openai.conversations.create()\n"
        "  response = openai.responses.create(\n"
        "      input='I prefer dark roast coffee in the mornings.',\n"
        "      conversation=conversation.id,\n"
        "      extra_body={'agent_reference': {'name': 'MemoryAgent', 'type': 'agent_reference'}},\n"
        "  )\n"
        "  # Memories are extracted after update_delay seconds of inactivity.\n"
        "\nStep 4: Session B (later) — recall the preference\n"
        "  new_conversation = openai.conversations.create()\n"
        "  recall_response = openai.responses.create(\n"
        "      input='Please order my usual coffee.',\n"
        "      conversation=new_conversation.id,\n"
        "      extra_body={'agent_reference': {'name': 'MemoryAgent', 'type': 'agent_reference'}},\n"
        "  )\n"
        "  # The agent uses stored memories to personalise the response.\n"
        "\nStep 5: Direct memory search via API\n"
        "  results = project_client.beta.memory_stores.search_memories(\n"
        "      name='demo-memory-store',\n"
        "      scope='demo_user_001',\n"
        "      items=[{'role': 'user', 'content': 'coffee preferences', 'type': 'message'}],\n"
        "      options=MemorySearchOptions(max_memories=5),\n"
        "  )\n"
    )
    print("=" * 60)
    print("Set FOUNDRY_PROJECT_ENDPOINT (and model names) to run live.\n")


# ---------------------------------------------------------------------------
# Live demo — only executed when credentials are configured
# ---------------------------------------------------------------------------

def run_live_demo(project: AIProjectClient) -> None:
    """Execute the full memory lifecycle against a real Foundry project."""
    openai = project.get_openai_client()

    # ── Step 1: Create Memory Store ────────────────────────────────────────
    print("\n[1] Creating memory store …")
    options = MemoryStoreDefaultOptions(
        chat_summary_enabled = True,
        user_profile_enabled = True,
        user_profile_details = (
            "Focus on user preferences such as beverage choices, working style, "
            "and tool preferences.  Avoid storing sensitive personal data."
        ),
    )
    definition = MemoryStoreDefaultDefinition(
        chat_model      = CHAT_MODEL,
        embedding_model = EMBEDDING_MODEL,
        options         = options,
    )
    memory_store = project.beta.memory_stores.create(
        name        = MEMORY_STORE_NAME,
        definition  = definition,
        description = "Demo memory store for agent memory sample",
    )
    print(f"    Memory store created: {memory_store.name}")

    # ── Step 2: Create Agent with Memory Tool ──────────────────────────────
    print("\n[2] Creating agent with memory search tool …")
    tool = MemorySearchPreviewTool(
        memory_store_name = MEMORY_STORE_NAME,
        scope             = USER_SCOPE,
        update_delay      = 5,  # seconds of inactivity before memory is written
    )
    agent = project.agents.create_version(
        agent_name = "MemoryDemoAgent",
        definition = PromptAgentDefinition(
            model        = CHAT_MODEL,
            instructions = (
                "You are a helpful personal assistant. "
                "Use your memory to recall user preferences from past conversations. "
                "When you remember something relevant, mention it naturally."
            ),
            tools = [tool],
        ),
    )
    print(f"    Agent created: {agent.name} (version {agent.version})")

    # ── Step 3: Session A — record preferences ─────────────────────────────
    print("\n[3] Session A — recording user preference …")
    conv_a = openai.conversations.create()
    resp_a = openai.responses.create(
        input       = "I always prefer dark roast coffee and I work best in the mornings.",
        conversation= conv_a.id,
        extra_body  = {"agent_reference": {"name": agent.name, "type": "agent_reference"}},
    )
    print(f"    Agent response: {resp_a.output_text}")
    print("    Waiting for memory extraction (update_delay) …")
    time.sleep(10)

    # ── Step 4: Session B — recall across session boundary ─────────────────
    print("\n[4] Session B (new conversation) — recalling stored memory …")
    conv_b = openai.conversations.create()
    resp_b = openai.responses.create(
        input       = "Please order my usual coffee for the morning meeting.",
        conversation= conv_b.id,
        extra_body  = {"agent_reference": {"name": agent.name, "type": "agent_reference"}},
    )
    print(f"    Agent response: {resp_b.output_text}")

    # ── Step 5: Direct memory search ──────────────────────────────────────
    print("\n[5] Direct memory search via API …")
    query = {"role": "user", "content": "coffee preferences", "type": "message"}
    results = project.beta.memory_stores.search_memories(
        name    = MEMORY_STORE_NAME,
        scope   = USER_SCOPE,
        items   = [query],
        options = MemorySearchOptions(max_memories=5),
    )
    print(f"    Found {len(results.memories)} memory item(s):")
    for mem in results.memories:
        print(f"      - {mem.memory_item.content}")

    # ── Step 6: Cleanup (optional) ─────────────────────────────────────────
    print("\n[6] Cleaning up — deleting memory store …")
    project.beta.memory_stores.delete_scope(name=MEMORY_STORE_NAME, scope=USER_SCOPE)
    project.beta.memory_stores.delete(MEMORY_STORE_NAME)
    print("    Memory store deleted.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    live = _require_env()
    if live:
        project_client = AIProjectClient(
            endpoint   = PROJECT_ENDPOINT,
            credential = DefaultAzureCredential(),
        )
        run_live_demo(project_client)
    else:
        _dry_run_demo()
