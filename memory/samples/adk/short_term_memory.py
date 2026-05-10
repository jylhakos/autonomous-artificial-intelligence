"""
ADK Short-Term Memory Sample
==============================
Demonstrates how Google Agent Development Kit (ADK) manages short-term memory
through Sessions and SessionState — ephemeral memory that lives only for the
duration of a single chat session.

Concepts demonstrated:
  - InMemorySessionService  : stores the active conversation event history
  - Session state           : key-value scratchpad updated by agent tools
  - State key templating    : inject live state values into agent prompts
  - State key prefixes      : user: / app: for cross-session persistence

References:
  https://adk.dev/sessions/session/
  https://cloud.google.com/blog/topics/developers-practitioners/remember-this-agent-state-and-memory-with-adk

Setup:
  1. Activate the virtual environment:
       source .venv/bin/activate        # Linux / macOS
       .venv\\Scripts\\activate           # Windows
  2. Install dependencies:
       pip install -r requirements.txt
  3. Export your Gemini API key:
       export GOOGLE_API_KEY="your-api-key-here"
  4. Run this script:
       python samples/adk/short_term_memory.py
"""

import asyncio
import os
from typing import Dict, Any

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Dependency guard — helpful error if ADK is not installed yet
# ---------------------------------------------------------------------------
try:
    from google.adk.agents import LlmAgent
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    from google.adk.tools.tool_context import ToolContext
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
APP_NAME = "short_term_memory_demo"
USER_ID  = "demo_user"
MODEL    = os.getenv("ADK_MODEL", "gemini-2.0-flash")

# A minimal quiz — two questions to show state tracking across turns
QUIZ = [
    ("What Python data structure maps keys to values?", "dictionary"),
    ("Which method returns all keys in a Python dictionary?",  "keys"),
]

# ---------------------------------------------------------------------------
# Agent tools — each tool reads/writes session state via ToolContext
# ---------------------------------------------------------------------------

def start_quiz(tool_context: ToolContext) -> Dict[str, Any]:
    """Initialise quiz state and return the first question."""
    state = tool_context.state
    state["quiz_started"]           = True
    state["current_question_index"] = 0
    state["correct_answers"]        = 0
    state["total_answered"]         = 0
    state["score_percentage"]       = 0

    if QUIZ:
        return {
            "status":           "started",
            "first_question":   QUIZ[0][0],
            "question_number":  1,
            "total_questions":  len(QUIZ),
        }
    return {"status": "error", "error_message": "No questions available"}


def submit_answer(tool_context: ToolContext, answer: str) -> Dict[str, Any]:
    """Evaluate the user's answer and advance to the next question."""
    state   = tool_context.state
    idx     = state.get("current_question_index", 0)

    if idx >= len(QUIZ):
        return {"status": "error", "error_message": "Quiz already complete"}

    correct_answer = QUIZ[idx][1]
    is_correct     = answer.strip().lower() == correct_answer.strip().lower()

    state["total_answered"] = state.get("total_answered", 0) + 1
    if is_correct:
        state["correct_answers"] = state.get("correct_answers", 0) + 1

    state["current_question_index"] = idx + 1

    total    = state["total_answered"]
    correct  = state["correct_answers"]
    state["score_percentage"] = int((correct / total) * 100) if total else 0

    next_idx = state["current_question_index"]
    if next_idx < len(QUIZ):
        return {
            "status":          "answered",
            "is_correct":      is_correct,
            "correct_answer":  correct_answer,
            "next_question":   QUIZ[next_idx][0],
            "question_number": next_idx + 1,
        }
    return {
        "status":          "quiz_complete",
        "is_correct":      is_correct,
        "correct_answer":  correct_answer,
        "final_score":     state["score_percentage"],
        "total_correct":   correct,
        "total_questions": len(QUIZ),
    }


def get_quiz_status(tool_context: ToolContext) -> Dict[str, Any]:
    """Return current quiz progress from session state."""
    state = tool_context.state
    return {
        "quiz_started":           state.get("quiz_started", False),
        "current_question_index": state.get("current_question_index", 0),
        "correct_answers":        state.get("correct_answers", 0),
        "total_answered":         state.get("total_answered", 0),
        "score_percentage":       state.get("score_percentage", 0),
    }

# ---------------------------------------------------------------------------
# Agent definition
# The prompt uses ADK key templating — {state_key} is injected automatically
# ---------------------------------------------------------------------------

AGENT_PROMPT = """
You are a helpful Python tutor.  Guide the user through a short quiz on Python
dictionaries.

CURRENT SESSION STATE (short-term memory):
  - Quiz started             : {quiz_started}
  - Current question index   : {current_question_index}
  - Correct answers          : {correct_answers}
  - Total answered           : {total_answered}
  - Score percentage         : {score_percentage}%

WORKFLOW:
1. Greet the user and ask if they are ready to start the quiz.
2. When they say yes, call start_quiz() to initialise state and get question 1.
3. Present each question clearly.
4. When the user provides an answer, call submit_answer(answer="<their answer>").
5. Give friendly feedback and continue until the quiz is complete.
6. Show the final score on completion.

Use get_quiz_status() whenever you need to check current progress.
"""

quiz_agent = LlmAgent(
    model       = MODEL,
    name        = "ShortTermMemoryAgent",
    instruction = AGENT_PROMPT,
    tools       = [start_quiz, submit_answer, get_quiz_status],
)

# ---------------------------------------------------------------------------
# Runner and session
# ---------------------------------------------------------------------------

async def run_demo() -> None:
    session_service = InMemorySessionService()
    runner          = Runner(
        agent           = quiz_agent,
        app_name        = APP_NAME,
        session_service = session_service,
    )

    session_id = "demo_session_001"
    await session_service.create_session(
        app_name   = APP_NAME,
        user_id    = USER_ID,
        session_id = session_id,
    )

    print("=" * 60)
    print("ADK Short-Term Memory Demo")
    print("=" * 60)
    print("Type your messages below.  Press Ctrl+C to exit.\n")

    # Simulated conversation turns for demonstration purposes
    demo_turns = [
        "Hello! I want to take the quiz.",
        "dictionary",
        "keys",
    ]

    for user_text in demo_turns:
        print(f"[User]  {user_text}")
        user_message = Content(parts=[Part(text=user_text)], role="user")

        response_text = "(no response)"
        async for event in runner.run_async(
            user_id    = USER_ID,
            session_id = session_id,
            new_message= user_message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                response_text = event.content.parts[0].text

        print(f"[Agent] {response_text}\n")

    # Show the final session state to illustrate short-term memory storage
    session = await session_service.get_session(
        app_name   = APP_NAME,
        user_id    = USER_ID,
        session_id = session_id,
    )
    print("-" * 60)
    print("Final session state (short-term memory scratchpad):")
    for key, value in session.state.items():
        print(f"  {key}: {value}")
    print("-" * 60)
    print("NOTE: This state is EPHEMERAL — it disappears when the session ends.")


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print(
            "WARNING: GOOGLE_API_KEY is not set.\n"
            "Export your Gemini API key before running:\n"
            "  export GOOGLE_API_KEY='your-api-key'\n"
        )
    asyncio.run(run_demo())
