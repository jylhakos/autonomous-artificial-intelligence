"""
AutoGen Hello Agent Example
============================
This is a simple example demonstrating how to create and run a basic AI agent
using Microsoft Agent Framework (AutoGen).

Prerequisites:
- Install: pip install -U "autogen-agentchat" "autogen-ext[openai]"
- Set environment variable: OPENAI_API_KEY

Reference: https://github.com/microsoft/agent-framework
"""

import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main() -> None:
    """
    Create a simple assistant agent and run a basic task.
    """
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("=" * 60)
    print("AutoGen Hello Agent Example")
    print("=" * 60)
    
    try:
        # Create an assistant agent with OpenAI model
        agent = AssistantAgent(
            "assistant", 
            OpenAIChatCompletionClient(model="gpt-4o-mini")
        )
        
        print("\n[Task] Say 'Hello World!'\n")
        
        # Run the agent with a simple task
        result = await agent.run(task="Say 'Hello World!'")
        print(f"\n[Agent Response]\n{result}\n")
        
        print("=" * 60)
        print("Example completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure OPENAI_API_KEY is set correctly")
        print("2. Check your OpenAI account has available credits")
        print("3. Verify network connectivity")


if __name__ == "__main__":
    asyncio.run(main())
