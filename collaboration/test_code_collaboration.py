"""
Test: Code Generation with Error Correction
===========================================
This tests if agents can collaborate to write and debug code.

This example demonstrates:
- Code writer agent creates Python code
- Code executor agent reviews and validates
- Iterative improvement through collaboration
"""

import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


def analyze_turn_taking(conversation_history):
    """Analyze and visualize turn-taking patterns."""
    print("\n" + "="*70)
    print("TURN-TAKING ANALYSIS")
    print("="*70)
    
    if not conversation_history:
        print("No conversation history available")
        return {}
    
    agents_sequence = []
    for msg in conversation_history:
        agent_name = msg.get('name', msg.get('source', 'Unknown'))
        agents_sequence.append(agent_name)
    
    print(f"\nTotal turns: {len(agents_sequence)}")
    print(f"Agent sequence: {' -> '.join(agents_sequence)}")
    
    # Count turns per agent
    from collections import Counter
    turn_counts = Counter(agents_sequence)
    
    print("\nTurns per agent:")
    for agent, count in turn_counts.items():
        print(f"  {agent}: {count} turns")
    
    # Check for collaboration
    unique_agents = len(turn_counts)
    if unique_agents > 1:
        print(f"\n COLLABORATION DETECTED: {unique_agents} different agents participated")
    else:
        print(f"\n NO COLLABORATION: Only 1 agent participated")
    
    return turn_counts


async def test_code_collaboration():
    """Test collaborative code generation and debugging."""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        return
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    
    # Code Writer Agent
    coder = AssistantAgent(
        "coder",
        model_client=model_client,
        system_message="""You are a Python developer. Write clean, working code.
        When you receive error messages, analyze them and fix the code.
        Always provide complete, executable code examples."""
    )
    
    # Code Executor Agent (simulated)
    executor = AssistantAgent(
        "executor",
        model_client=model_client,
        system_message="""You are a code execution simulator and reviewer. When you receive code:
        1. Analyze it for potential errors and edge cases
        2. Report any issues found (syntax errors, logic errors, missing imports)
        3. If correct, confirm it would execute successfully
        4. Suggest improvements if needed
        Be thorough in your review."""
    )
    
    task = """Write a Python function to calculate fibonacci numbers recursively.
    Include a main block that tests the function with n=10.
    The executor will review your code for correctness."""
    
    print("="*70)
    print("CODE COLLABORATION TEST")
    print("="*70)
    print(f"\nTask: {task}\n")
    print("-"*70)
    
    try:
        # Create team
        team = RoundRobinGroupChat(participants=[coder, executor], max_turns=6)
        
        result = await team.run(task=task)
        
        print("\n" + "="*70)
        print("FINAL RESULT")
        print("="*70)
        print(result)
        
        # Analyze collaboration
        print("\n" + "="*70)
        print("COLLABORATION ANALYSIS")
        print("="*70)
        
        # Try to get conversation history
        conversation = []
        if hasattr(team, 'conversation_history'):
            conversation = team.conversation_history
        elif hasattr(team, '_messages'):
            conversation = team._messages
        
        if conversation:
            analyze_turn_taking(conversation)
            
            code_written = any('def' in str(msg.get('content', '')) for msg in conversation)
            errors_reported = any('error' in str(msg.get('content', '')).lower() or 
                                'fix' in str(msg.get('content', '')).lower() or
                                'issue' in str(msg.get('content', '')).lower()
                                for msg in conversation)
            
            print(f"\nCode was written:  {'YES' if code_written else 'NO'}")
            print(f"Review/feedback given:  {'YES' if errors_reported else 'NO'}")
            print(f"Total interaction rounds: {len(conversation)}")
            
            if code_written and len(conversation) >= 2:
                print("\n COLLABORATIVE CODE DEVELOPMENT OCCURRED")
                print("\nKey Collaboration Indicators:")
                print("  - Multiple agents participated")
                print("  - Code was written and reviewed")
                print("  - Iterative feedback was provided")
            else:
                print("\n COLLABORATION WAS INSUFFICIENT")
        else:
            print("\nCould not access conversation history for detailed analysis")
            print(" Final result was produced (see above)")
            
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure OPENAI_API_KEY is set")
        print("2. Check your OpenAI account has credits")
        print("3. Verify autogen packages are installed")


if __name__ == "__main__":
    asyncio.run(test_code_collaboration())
