"""
AutoGen Multi-Agent Collaboration Example
==========================================
This example demonstrates collaborative AI agents working together to solve
a complex task. Multiple specialized agents communicate, share context, and
divide work to achieve a common goal.

Agent Roles:
- Planner: Breaks down the task into subtasks
- Researcher: Gathers information and facts
- Writer: Composes structured content
- Critic: Reviews and provides feedback

Prerequisites:
- Install: pip install -U "autogen-agentchat" "autogen-ext[openai]"
- Set environment variable: OPENAI_API_KEY

Reference: https://github.com/microsoft/agent-framework/tree/main/python/samples
"""

import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main() -> None:
    """
    Demonstrate multi-agent collaboration with specialized roles.
    """
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("=" * 70)
    print("AutoGen Multi-Agent Collaboration Example")
    print("=" * 70)
    
    try:
        # Create the OpenAI client
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
        
        # Create specialized agents
        print("\n[Creating Specialized Agents]")
        print("- Planner: Task decomposition and coordination")
        print("- Researcher: Information gathering")
        print("- Writer: Content composition")
        print("- Critic: Quality review and feedback\n")
        
        # Planner Agent - Breaks down complex tasks
        planner = AssistantAgent(
            "planner",
            model_client=model_client,
            system_message="""You are a planning agent. Your role is to:
            1. Analyze the given task
            2. Break it down into logical subtasks
            3. Coordinate with other agents
            4. Ensure all subtasks are addressed
            Be concise and structured in your planning."""
        )
        
        # Researcher Agent - Gathers information
        researcher = AssistantAgent(
            "researcher",
            model_client=model_client,
            system_message="""You are a research agent. Your role is to:
            1. Gather relevant information and facts
            2. Provide accurate data and context
            3. Answer questions with evidence
            Focus on accuracy and relevance."""
        )
        
        # Writer Agent - Composes content
        writer = AssistantAgent(
            "writer",
            model_client=model_client,
            system_message="""You are a writing agent. Your role is to:
            1. Compose clear, structured content
            2. Synthesize information from researchers
            3. Create well-formatted outputs
            Write professionally and concisely."""
        )
        
        # Critic Agent - Reviews and provides feedback
        critic = AssistantAgent(
            "critic",
            model_client=model_client,
            system_message="""You are a critic agent. Your role is to:
            1. Review outputs from other agents
            2. Identify issues, gaps, or improvements
            3. Provide constructive feedback
            4. Validate quality and accuracy
            Be thorough but fair in your critique."""
        )
        
        # Create a team with round-robin communication
        team = RoundRobinGroupChat(
            participants=[planner, researcher, writer, critic],
            max_turns=8
        )
        
        # Define the collaborative task
        task = """Create a brief explanation (3-4 sentences) of how collaborative 
        AI agents work in autonomous systems. Include the key benefits of using 
        multiple specialized agents versus a single general-purpose agent."""
        
        print(f"[Task]\n{task}\n")
        print("-" * 70)
        print("[Agent Collaboration in Progress...]\n")
        
        # Run the multi-agent collaboration
        result = await team.run(task=task)
        
        print("\n" + "-" * 70)
        print("[Final Output]")
        print("-" * 70)
        print(result)
        
        print("\n" + "=" * 70)
        print("Multi-Agent Collaboration Completed Successfully!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("✓ Agents specialized in different roles (planning, research, writing, review)")
        print("✓ Agents communicated and shared context through the team")
        print("✓ Work was distributed efficiently across agents")
        print("✓ Output was refined through iterative feedback")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure OPENAI_API_KEY is set correctly")
        print("2. Check your OpenAI account has available credits")
        print("3. Verify the autogen packages are installed")
        print("   pip install -U 'autogen-agentchat' 'autogen-ext[openai]'")


if __name__ == "__main__":
    asyncio.run(main())
