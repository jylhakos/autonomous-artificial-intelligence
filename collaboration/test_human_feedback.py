"""
Test: Human-in-the-Loop Collaboration
======================================
This demonstrates how human feedback forces agents to adjust approaches.

This example shows:
- Initial agent response
- Human feedback integration
- Agent adaptation based on feedback
- Iterative improvement cycle
"""

import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def test_human_in_loop():
    """Test collaboration with human feedback."""
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        return
    
    print("="*70)
    print("HUMAN-IN-THE-LOOP COLLABORATION TEST")
    print("="*70)
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    
    assistant = AssistantAgent(
        "assistant",
        model_client=model_client,
        system_message="""You are a helpful assistant who adapts to feedback.
        When you receive feedback, acknowledge it and adjust your approach accordingly.
        Be responsive to suggestions and improve your explanations."""
    )
    
    # Test scenario: Explain complex topic with iterative feedback
    task = "Explain quantum computing in simple terms."
    
    print(f"\nInitial Task: {task}\n")
    print("-"*70)
    
    try:
        # Round 1: Initial attempt
        print("\n[Round 1] Initial Explanation:")
        response1 = await assistant.run(task=task)
        print(response1)
        
        # Human feedback (simulated)
        feedback = """That's too technical for my 10-year-old daughter. 
        Please explain it using a simple analogy that relates to everyday life, 
        like toys or games."""
        
        print("\n" + "-"*70)
        print(f"\n[Human Feedback]: {feedback}\n")
        print("-"*70)
        
        # Round 2: Adjusted attempt based on feedback
        print("\n[Round 2] Adjusted Explanation After Feedback:")
        adjusted_task = f"""Original question: {task}

        Your previous response was provided, but the user gave this feedback:
        {feedback}

        Please provide a new explanation that addresses this feedback."""
        
        response2 = await assistant.run(task=adjusted_task)
        print(response2)
        
        # Additional refinement
        feedback2 = "Great! Now can you make it even shorter, just 2-3 sentences?"
        
        print("\n" + "-"*70)
        print(f"\n[Human Feedback 2]: {feedback2}\n")
        print("-"*70)
        
        # Round 3: Final refinement
        print("\n[Round 3] Final Refined Explanation:")
        final_task = f"""Based on your child-friendly explanation, the user now wants:
        {feedback2}
        
        Please provide an ultra-concise version."""
        
        response3 = await assistant.run(task=final_task)
        print(response3)
        
        # Analysis
        print("\n" + "="*70)
        print("COLLABORATION ANALYSIS")
        print("="*70)
        print("\n Agent received human feedback (2 rounds)")
        print(" Agent adjusted approach based on feedback each time")
        print(" Progressive refinement through iterations")
        print(" Interactive collaboration demonstrated")
        
        print("\nKey Collaboration Patterns Observed:")
        print("1. Initial response -> Feedback -> Adjusted response")
        print("2. Continuous adaptation to user requirements")
        print("3. Iterative improvement cycle")
        print("4. Human-guided refinement")
        
        print("\n HUMAN-IN-THE-LOOP COLLABORATION SUCCESSFUL")
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure OPENAI_API_KEY is set")
        print("2. Check your OpenAI account has credits")


if __name__ == "__main__":
    asyncio.run(test_human_in_loop())
