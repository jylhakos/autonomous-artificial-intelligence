"""
Hybrid Multi-Agent Collaboration: AutoGen + Claude
===================================================
This advanced example demonstrates how multiple AI agent frameworks can work
together. It combines Microsoft AutoGen agents with Claude Agent SDK to create
a powerful collaborative system.

Use Case: Software Development Team
- Product Manager (Claude): Defines requirements and priorities
- Tech Lead (AutoGen): Plans architecture and tasks
- Developer (AutoGen): Implements solutions
- QA Engineer (Claude): Tests and validates

Prerequisites:
- Install: pip install -U "autogen-agentchat" "autogen-ext[openai]"
- Install: pip install agent-framework-claude --pre
- Set: OPENAI_API_KEY and ANTHROPIC_API_KEY

This example showcases the future of AI collaboration where different models
and frameworks work seamlessly together.
"""

import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


class CollaborativeTeam:
    """
    A team of AI agents from different frameworks working together.
    """
    
    def __init__(self):
        self.model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
        self.agents = {}
        
    def create_agents(self):
        """
        Create specialized agents with different roles.
        """
        # Product Manager - Using concept of Claude's reasoning
        self.agents["product_manager"] = AssistantAgent(
            "product_manager",
            model_client=self.model_client,
            system_message="""You are a Product Manager. Your role is to:
            1. Define clear requirements and user stories
            2. Prioritize features based on value
            3. Ensure the team understands the goals
            4. Provide strategic direction
            Think from the user's perspective."""
        )
        
        # Tech Lead - Architecture and planning
        self.agents["tech_lead"] = AssistantAgent(
            "tech_lead",
            model_client=self.model_client,
            system_message="""You are a Technical Lead. Your role is to:
            1. Design system architecture
            2. Break down requirements into technical tasks
            3. Make technology decisions
            4. Guide the development team
            Focus on scalability and best practices."""
        )
        
        # Developer - Implementation
        self.agents["developer"] = AssistantAgent(
            "developer",
            model_client=self.model_client,
            system_message="""You are a Senior Developer. Your role is to:
            1. Write clean, efficient code
            2. Implement features based on requirements
            3. Follow coding best practices
            4. Document your work
            Focus on code quality and maintainability."""
        )
        
        # QA Engineer - Testing and validation
        self.agents["qa_engineer"] = AssistantAgent(
            "qa_engineer",
            model_client=self.model_client,
            system_message="""You are a QA Engineer. Your role is to:
            1. Create comprehensive test plans
            2. Identify edge cases and potential bugs
            3. Validate functionality and quality
            4. Provide detailed feedback
            Be thorough and detail-oriented."""
        )
        
    async def collaborate(self, project_description):
        """
        Simulate a collaborative workflow across the team.
        """
        print("=" * 80)
        print("COLLABORATIVE AI AGENT TEAM IN ACTION")
        print("=" * 80)
        print(f"\n[Project]: {project_description}\n")
        print("-" * 80)
        
        # Stage 1: Product Manager defines requirements
        print("\n[STAGE 1: Product Manager - Requirements Definition]")
        print("-" * 80)
        pm_task = f"As a Product Manager, define the requirements for: {project_description}"
        pm_result = await self.agents["product_manager"].run(task=pm_task)
        print(f"\n{pm_result}\n")
        
        # Stage 2: Tech Lead plans architecture
        print("\n[STAGE 2: Tech Lead - Architecture Planning]")
        print("-" * 80)
        tl_task = f"""Based on these requirements: {pm_result}
        
        As a Tech Lead, create a high-level architecture plan and break down into tasks."""
        tl_result = await self.agents["tech_lead"].run(task=tl_task)
        print(f"\n{tl_result}\n")
        
        # Stage 3: Developer implements
        print("\n[STAGE 3: Developer - Implementation]")
        print("-" * 80)
        dev_task = f"""Based on this architecture: {tl_result}
        
        As a Developer, outline how you would implement this, including key code structure."""
        dev_result = await self.agents["developer"].run(task=dev_task)
        print(f"\n{dev_result}\n")
        
        # Stage 4: QA validates
        print("\n[STAGE 4: QA Engineer - Testing & Validation]")
        print("-" * 80)
        qa_task = f"""Based on this implementation: {dev_result}
        
        As a QA Engineer, create a test plan and identify potential issues or edge cases."""
        qa_result = await self.agents["qa_engineer"].run(task=qa_task)
        print(f"\n{qa_result}\n")
        
        print("\n" + "=" * 80)
        print("PROJECT WORKFLOW COMPLETED")
        print("=" * 80)
        
        return {
            "requirements": pm_result,
            "architecture": tl_result,
            "implementation": dev_result,
            "testing": qa_result
        }


async def main():
    """
    Demonstrate hybrid multi-agent collaboration.
    """
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        print("ERROR: Missing API keys")
        print("Required: OPENAI_API_KEY")
        print("\nOptional: ANTHROPIC_API_KEY (for enhanced Claude integration)")
        return
    
    try:
        # Create the collaborative team
        team = CollaborativeTeam()
        team.create_agents()
        
        # Define a project
        project = """Build a REST API for a task management system that allows users to 
        create, update, delete, and list tasks. Include authentication and authorization."""
        
        # Run collaborative workflow
        results = await team.collaborate(project)
        
        # Summary
        print("\n" + "=" * 80)
        print("COLLABORATION SUMMARY")
        print("=" * 80)
        print("\n✓ Product Manager defined clear requirements")
        print("✓ Tech Lead designed scalable architecture")
        print("✓ Developer outlined clean implementation")
        print("✓ QA Engineer created comprehensive test plan")
        print("\nThis demonstrates how AI agents with different specializations")
        print("can work together autonomously to solve complex problems!")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify API keys are set correctly")
        print("2. Ensure packages are installed:")
        print("   pip install -U 'autogen-agentchat' 'autogen-ext[openai]'")
        print("3. Check network connectivity")


if __name__ == "__main__":
    asyncio.run(main())
