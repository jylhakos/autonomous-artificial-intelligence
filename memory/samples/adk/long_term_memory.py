"""
ADK Long-Term Memory Sample
==============================
Demonstrates how Google Agent Development Kit (ADK) implements long-term memory
using MemoryService — a searchable archive of knowledge extracted from past
sessions that persists across conversation boundaries.

Concepts demonstrated:
  - InMemoryMemoryService   : in-process long-term store (good for prototyping)
  - add_session_to_memory() : ingest a completed session into the knowledge store
  - load_memory tool        : agent-callable tool for semantic memory retrieval
  - PreloadMemoryTool       : automatically prepend memories to every agent turn
  - Two-agent pipeline      : one agent writes context, another recalls it

Long-term memory tiers in ADK:
  ┌──────────────────────────────┬───────────────────────────────┐
  │ InMemoryMemoryService        │ VertexAiMemoryBankService     │
  │ (local / prototyping)        │ (production / persistent)     │
  ├──────────────────────────────┼───────────────────────────────┤
  │ No setup required            │ Requires GCP project +        │
  │ Lost on process restart      │ Agent Engine instance         │
  │ Keyword search               │ Semantic / vector search      │
  └──────────────────────────────┴───────────────────────────────┘

References:
  https://adk.dev/sessions/memory/
  https://cloud.google.com/blog/topics/developers-practitioners/remember-this-agent-state-and-memory-with-adk
  https://codelabs.developers.google.com/adkcourse/instructions#7

Setup:
  1. Activate the virtual environment:
       source .venv/bin/activate        # Linux / macOS
       .venv\\Scripts\\activate           # Windows
  2. Install dependencies:
       pip install -r requirements.txt
  3. Export your Gemini API key:
       export GOOGLE_API_KEY="your-api-key-here"
  4. Run this script:
       python samples/adk/long_term_memory.py
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------
try:
    from google.adk.agents import LlmAgent
    from google.adk.sessions import InMemorySessionService
    from google.adk.memory import InMemoryMemoryService
    from google.adk.runners import Runner
    from google.adk.tools import load_memory
    from google.genai.types import Content, Part
except ImportError as exc:
    raise SystemExit(
        "google-adk is not installed.  "
        "Activate the virtual environment and run:\n"
        "  pip install -r requirements.txt"
    ) from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_NAME = "long_term_memory_demo"
USER_ID  = "returning_user"
MODEL    = os.getenv("ADK_MODEL", "gemini-2.0-flash")

# ---------------------------------------------------------------------------
# Agent 1 — captures facts stated by the user
# ---------------------------------------------------------------------------
info_capture_agent = LlmAgent(
    model       = MODEL,
    name        = "InfoCaptureAgent",
    instruction = (
        "You are a helpful note-taking assistant.  "
        "Acknowledge everything the user tells you warmly and concisely."
    ),
)

# ---------------------------------------------------------------------------
# Agent 2 — answers questions by searching long-term memory
# The built-in 'load_memory' tool lets the agent query the MemoryService.
# ---------------------------------------------------------------------------
memory_recall_agent = LlmAgent(
    model       = MODEL,
    name        = "MemoryRecallAgent",
    instruction = (
        "You are a helpful assistant with access to the user's past conversations.  "
        "When the user asks a question, use the 'load_memory' tool to search for "
        "relevant information from previous sessions before answering.  "
        "If memory is found, reference it explicitly in your answer."
    ),
    tools = [load_memory],
)

# ---------------------------------------------------------------------------
# Shared services (must be the same instances for both runners)
# ---------------------------------------------------------------------------
session_service = InMemorySessionService()
memory_service  = InMemoryMemoryService()   # swap for VertexAiMemoryBankService in production


async def run_session(
    runner:         Runner,
    session_id:     str,
    user_text:      str,
) -> str:
    """Run a single user turn and return the agent's response text."""
    await session_service.create_session(
        app_name   = APP_NAME,
        user_id    = USER_ID,
        session_id = session_id,
    )
    message = Content(parts=[Part(text=user_text)], role="user")
    response_text = "(no response)"

    async for event in runner.run_async(
        user_id     = USER_ID,
        session_id  = session_id,
        new_message = message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text

    return response_text


async def run_demo() -> None:
    print("=" * 60)
    print("ADK Long-Term Memory Demo")
    print("=" * 60)

    # ── Phase 1: Record user information in Session A ──────────────────────
    print("\n[Phase 1] Capturing facts into Session A …")

    runner_capture = Runner(
        agent           = info_capture_agent,
        app_name        = APP_NAME,
        session_service = session_service,
        memory_service  = memory_service,
    )

    facts = [
        "My favourite programming language is Python.",
        "I prefer dark-mode editors and always use single quotes in Python.",
        "I am currently learning about AI agent memory systems.",
    ]

    for idx, fact in enumerate(facts):
        session_id    = f"session_capture_{idx}"
        response_text = await run_session(runner_capture, session_id, fact)
        print(f"  [User]  {fact}")
        print(f"  [Agent] {response_text}\n")

        # Ingest each completed session into the long-term knowledge store
        completed = await session_service.get_session(
            app_name   = APP_NAME,
            user_id    = USER_ID,
            session_id = session_id,
        )
        await memory_service.add_session_to_memory(completed)
        print(f"  >> Session '{session_id}' added to long-term memory store.\n")

    # ── Phase 2: Start a brand-new session — ask questions ─────────────────
    print("-" * 60)
    print("[Phase 2] New session — recalling stored memories …\n")

    runner_recall = Runner(
        agent           = memory_recall_agent,
        app_name        = APP_NAME,
        session_service = session_service,
        memory_service  = memory_service,
    )

    questions = [
        "What is my favourite programming language?",
        "What do you know about my editor preferences?",
        "What topic am I currently studying?",
    ]

    for idx, question in enumerate(questions):
        session_id    = f"session_recall_{idx}"
        response_text = await run_session(runner_recall, session_id, question)
        print(f"  [User]  {question}")
        print(f"  [Agent] {response_text}\n")

    print("=" * 60)
    print("Demo complete.")
    print(
        "\nKey takeaway:\n"
        "  The recall agent answered questions from a completely fresh session\n"
        "  by querying the InMemoryMemoryService that was populated in Phase 1.\n"
        "  Replace InMemoryMemoryService with VertexAiMemoryBankService for\n"
        "  persistent, cloud-hosted long-term memory in production.\n"
    )


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print(
            "WARNING: GOOGLE_API_KEY is not set.\n"
            "Export your Gemini API key before running:\n"
            "  export GOOGLE_API_KEY='your-api-key'\n"
        )
    asyncio.run(run_demo())
