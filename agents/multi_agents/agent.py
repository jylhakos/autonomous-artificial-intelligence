"""
Multi-Agent System Example — Google ADK (Supervisor Pattern)
=============================================================
A multi-agent system divides responsibilities across multiple specialized agents
coordinated by a supervisor (orchestrator). The supervisor receives a high-level
goal, breaks it into sub-tasks, and delegates each to the right specialist.

Architecture used here — Supervisor / Hierarchical pattern:

    SupervisorAgent
    ├── ResearchAgent     (fact-finding)
    ├── AnalysisAgent     (trend analysis)
    ├── WriterAgent       (content generation)
    └── ReviewerAgent     (quality gate / judge)

The supervisor uses LLM-driven delegation: it reads the conversation context,
decides which sub-agent to activate next, and synthesises the final output.
This mirrors the orchestrator-worker pattern described in:
  https://cloud.google.com/blog/topics/developers-practitioners/
        building-collaborative-ai-a-developers-guide-to-multi-agent-systems-with-adk

When to use multi-agent over single-agent (Microsoft CAF guidance):
  1. Tasks cross security or compliance boundaries.
  2. Multiple teams own separate knowledge domains.
  3. Future growth will require independent scaling per domain.
  See: https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/ai-agents/
             single-agent-multiple-agents

Usage:
    python agent.py
"""

import asyncio

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

load_dotenv()

# ---------------------------------------------------------------------------
# Specialist sub-agents  (each has a narrow, well-defined responsibility)
# ---------------------------------------------------------------------------
research_agent = LlmAgent(
    name="ResearchAgent",
    model="gemini-2.0-flash-exp",
    instruction=(
        "You are a senior researcher. Given the topic in the conversation, "
        "produce 5 concrete, cited-style facts with sources or publication years "
        "where possible. Output as a numbered list."
    ),
    description="Gathers factual evidence on a given topic.",
)

analysis_agent = LlmAgent(
    name="AnalysisAgent",
    model="gemini-2.0-flash-exp",
    instruction=(
        "You are a data analyst. Based on the research facts already in the "
        "conversation, identify 3 key trends and their potential implications. "
        "Output as a structured analysis with a 'Trend' and 'Implication' for each."
    ),
    description="Analyses facts and extracts trends.",
)

writer_agent = LlmAgent(
    name="WriterAgent",
    model="gemini-2.0-flash-exp",
    instruction=(
        "You are a technical writer. Using the research facts and the trend "
        "analysis already in the conversation, write a 200-word executive summary "
        "suitable for a non-technical leadership audience."
    ),
    description="Produces polished written summaries.",
)

reviewer_agent = LlmAgent(
    name="ReviewerAgent",
    model="gemini-2.0-flash-exp",
    instruction=(
        "You are a critical reviewer (the 'judge' agent). Evaluate the executive "
        "summary in the conversation against these criteria: accuracy, clarity, "
        "completeness, and professional tone. "
        "If the summary meets all criteria, output: APPROVED: <final summary>. "
        "If it needs changes, output: REVISION NEEDED: <specific feedback>."
    ),
    description="Quality-gates the final output before delivery.",
)

# ---------------------------------------------------------------------------
# Supervisor orchestrator
# A SequentialAgent is used here so each specialist runs in order, with full
# conversation context passed between them via shared session state.
# For dynamic routing (LLM decides order), replace with an LlmAgent that
# lists the specialists as sub_agents and instructs it to delegate.
# ---------------------------------------------------------------------------
supervisor = SequentialAgent(
    name="SupervisorAgent",
    sub_agents=[research_agent, analysis_agent, writer_agent, reviewer_agent],
    description=(
        "Orchestrates research, analysis, writing, and review specialists "
        "in sequence to produce a quality-assured executive summary."
    ),
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def run_multi_agent(prompt: str) -> None:
    session_service = InMemorySessionService()
    runner = Runner(
        agent=supervisor,
        app_name="multi_agent_demo",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="multi_agent_demo",
        user_id="user_01",
    )

    print(f"\n[SupervisorAgent] Prompt: {prompt}\n")
    print("=" * 70)

    async for event in runner.run_async(
        user_id="user_01",
        session_id=session.id,
        new_message={"role": "user", "parts": [{"text": prompt}]},
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(
        run_multi_agent(
            "Produce an executive summary on the current state and future outlook "
            "of generative AI in enterprise software development."
        )
    )
