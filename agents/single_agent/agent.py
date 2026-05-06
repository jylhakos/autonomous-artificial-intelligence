"""
Single-Agent Example — Google ADK
==================================
A single agent is a foundational LlmAgent that performs one specific task
based on its instruction prompt. It handles the full reasoning loop — tool
calls included — within a single entity.

Usage:
    python agent.py
"""

import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

load_dotenv()

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------
# A single-agent is given a clear, bounded instruction. It answers or acts
# entirely on its own without delegating to any other agent.
researcher = LlmAgent(
    name="Researcher",
    model="gemini-2.0-flash-exp",
    instruction=(
        "You are a concise research assistant. "
        "When given a topic, provide a short factual summary (3-5 sentences) "
        "covering the key concepts, current trends, and practical applications."
    ),
    description="A research agent that produces concise topic summaries.",
)

# ---------------------------------------------------------------------------
# Runner — thin wrapper that manages session state and event loops
# ---------------------------------------------------------------------------
session_service = InMemorySessionService()
runner = Runner(
    agent=researcher,
    app_name="single_agent_demo",
    session_service=session_service,
)


async def run_agent(prompt: str) -> None:
    """Send a prompt to the single agent and print its response."""
    session = await session_service.create_session(
        app_name="single_agent_demo",
        user_id="user_01",
    )

    print(f"\nPrompt: {prompt}\n")
    print("-" * 60)

    async for event in runner.run_async(
        user_id="user_01",
        session_id=session.id,
        new_message={"role": "user", "parts": [{"text": prompt}]},
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

    print("-" * 60)


if __name__ == "__main__":
    # Example prompt — edit freely
    topic = "renewable energy trends in 2025"
    asyncio.run(run_agent(f"Research the latest trends in {topic}."))
