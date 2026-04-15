"""
Claude Agent SDK Basic Example
===============================
This example demonstrates how to use Claude Agent SDK with Microsoft Agent Framework.
Claude's advanced reasoning capabilities combined with agent orchestration.

Prerequisites:
- Install: pip install agent-framework-claude --pre
- Set environment variable: ANTHROPIC_API_KEY

References:
- https://github.com/anthropics/claude-agent-sdk-python
- https://devblogs.microsoft.com/agent-framework/build-ai-agents-with-claude-agent-sdk-and-microsoft-agent-framework/
"""

import asyncio
import os
from agent_framework_claude import ClaudeAgent


async def basic_example():
    """
    Basic Claude Agent example - simple question and answer.
    """
    print("=" * 70)
    print("Claude Agent SDK - Basic Example")
    print("=" * 70)
    
    try:
        async with ClaudeAgent(
            instructions="You are a helpful assistant specializing in AI and technology.",
        ) as agent:
            print("\n[Question] What is Microsoft Agent Framework?\n")
            
            response = await agent.run("What is Microsoft Agent Framework?")
            print(f"[Claude's Response]\n{response.text}\n")
            
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("\nPlease ensure ANTHROPIC_API_KEY is set:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")


async def coding_assistant_example():
    """
    Claude Agent with tools - demonstrates file system and bash operations.
    """
    print("=" * 70)
    print("Claude Agent SDK - Coding Assistant with Tools")
    print("=" * 70)
    
    try:
        async with ClaudeAgent(
            instructions="You are a helpful coding assistant.",
            tools=["Read", "Write", "Bash", "Glob"],
        ) as agent:
            print("\n[Task] List all Python files in the current directory\n")
            
            response = await agent.run("List all Python files in the current directory")
            print(f"[Claude's Response]\n{response.text}\n")
            
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("\nPlease ensure ANTHROPIC_API_KEY is set:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")


async def main():
    """
    Run both Claude Agent examples.
    """
    # Check if API key is set
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Please set ANTHROPIC_API_KEY environment variable")
        print("Example: export ANTHROPIC_API_KEY='your-api-key-here'")
        print("\nGet your API key from: https://console.anthropic.com/")
        return
    
    print("\nRunning Claude Agent SDK Examples...\n")
    
    # Run basic example
    await basic_example()
    
    print("\n" + "=" * 70 + "\n")
    
    # Run coding assistant example
    await coding_assistant_example()
    
    print("\n" + "=" * 70)
    print("All Claude Agent Examples Completed Successfully!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("✓ Basic agent interaction with instructions")
    print("✓ Agent with tools (Read, Write, Bash, Glob)")
    print("✓ Advanced reasoning with Claude models")
    print("✓ Integration with Microsoft Agent Framework")


if __name__ == "__main__":
    asyncio.run(main())
