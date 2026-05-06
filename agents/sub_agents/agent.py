"""
Sub-Agents Example — Google ADK
=================================
Sub-agents are specialized LlmAgents managed by a parent (orchestrator) agent.
They operate within the parent's session context, sharing state and conversation
history. Two orchestration modes are shown:

  1. SequentialAgent  — runs sub-agents one after another (pipeline / chain)
  2. ParallelAgent    — runs sub-agents concurrently (fan-out)

Key distinctions from "agents as tools":
  - Sub-agents share the session state of the parent.
  - Sub-agents handle complex, stateful, multi-step processes.
  - Agents-as-tools are stateless, isolated, and reusable across agents.

Usage:
    python agent.py
"""

import asyncio

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

load_dotenv()

# ---------------------------------------------------------------------------
# Sub-agent definitions
# Each sub-agent has a focused instruction (bounded responsibility).
# ---------------------------------------------------------------------------
research_agent = LlmAgent(
    name="Researcher",
    model="gemini-2.0-flash-exp",
    instruction=(
        "Extract 3-5 key facts about the topic provided in the conversation. "
        "Be concise and factual. Output as a numbered list."
    ),
    description="Extracts structured facts about a given topic.",
)

writer_agent = LlmAgent(
    name="Writer",
    model="gemini-2.0-flash-exp",
    instruction=(
        "Using the research facts already in the conversation, write a short "
        "professional blog-style paragraph (4-6 sentences) suitable for a "
        "general technical audience."
    ),
    description="Writes a professional summary based on research output.",
)

editor_agent = LlmAgent(
    name="Editor",
    model="gemini-2.0-flash-exp",
    instruction=(
        "Review the blog paragraph in the conversation for clarity, grammar, "
        "and tone. Output only the final improved version."
    ),
    description="Edits and polishes written content.",
)

# ---------------------------------------------------------------------------
# Parallel sub-agents — independent tasks run concurrently
# ---------------------------------------------------------------------------
flight_finder = LlmAgent(
    name="FlightFinder",
    model="gemini-2.0-flash-exp",
    instruction="Suggest 2-3 realistic flight options (airline, rough price) for the trip described.",
    description="Finds flight options.",
)

hotel_finder = LlmAgent(
    name="HotelFinder",
    model="gemini-2.0-flash-exp",
    instruction="Suggest 2-3 realistic hotel options (name, neighbourhood, rough price) for the trip described.",
    description="Finds hotel options.",
)

# ---------------------------------------------------------------------------
# Orchestrator 1: SequentialAgent (Researcher → Writer → Editor)
# Output of each sub-agent feeds into the next via shared session state.
# ---------------------------------------------------------------------------
blog_team = SequentialAgent(
    name="BlogTeam",
    sub_agents=[research_agent, writer_agent, editor_agent],
    description="Orchestrates a research-write-edit pipeline sequentially.",
)

# ---------------------------------------------------------------------------
# Orchestrator 2: ParallelAgent (FlightFinder + HotelFinder run at the same time)
# Useful when sub-tasks are independent and results are aggregated afterwards.
# ---------------------------------------------------------------------------
travel_planner = ParallelAgent(
    name="TravelPlanner",
    sub_agents=[flight_finder, hotel_finder],
    description="Gathers flight and hotel data simultaneously.",
)


# ---------------------------------------------------------------------------
# Helper: run an orchestrator with a user prompt
# ---------------------------------------------------------------------------
async def run_orchestrator(orchestrator, app_name: str, prompt: str) -> None:
    session_service = InMemorySessionService()
    runner = Runner(
        agent=orchestrator,
        app_name=app_name,
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name=app_name,
        user_id="user_01",
    )

    print(f"\n[{orchestrator.name}] Prompt: {prompt}\n")
    print("=" * 60)

    async for event in runner.run_async(
        user_id="user_01",
        session_id=session.id,
        new_message={"role": "user", "parts": [{"text": prompt}]},
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

    print("=" * 60)


if __name__ == "__main__":
    # --- Demo 1: Sequential pipeline ---
    asyncio.run(
        run_orchestrator(
            blog_team,
            app_name="blog_team_demo",
            prompt="Write a blog post about the impact of quantum computing on cryptography.",
        )
    )

    # --- Demo 2: Parallel fan-out ---
    asyncio.run(
        run_orchestrator(
            travel_planner,
            app_name="travel_planner_demo",
            prompt="Plan a 5-day trip to Tokyo in April for two people with a mid-range budget.",
        )
    )
