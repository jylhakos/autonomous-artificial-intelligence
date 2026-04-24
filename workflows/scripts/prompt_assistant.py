#!/usr/bin/env python3
"""
Prompt Assistant for Vibe Coding with Foundry Local

This tool helps developers create optimized prompts for AI-assisted coding by:
1. Analyzing user intent and extracting requirements
2. Generating structured, comprehensive prompts
3. Using Foundry Local for local AI processing (privacy-focused)
4. Supporting interactive refinement workflows

Usage:
    Interactive mode:  python prompt_assistant.py
    CLI mode:          python prompt_assistant.py "Create a REST API for..."
"""

import os
import sys
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
import requests

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not found. Install with: pip install openai")
    sys.exit(1)


class PromptCategory(Enum):
    """Categories of development prompts."""
    CODE_GENERATION = "code_generation"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    DATA_PIPELINE = "data_pipeline"
    UI_DESIGN = "ui_design"


@dataclass
class PromptTemplate:
    """Structured prompt template for vibe coding."""
    category: str
    intent: str
    context: str
    requirements: List[str]
    constraints: List[str]
    expected_output: str
    
    def to_prompt(self) -> str:
        """Convert template to formatted prompt string."""
        prompt_parts = [
            f"Task: {self.intent}\n",
            f"\nContext:\n{self.context}\n",
            f"\nRequirements:",
        ]
        
        for req in self.requirements:
            prompt_parts.append(f"- {req}")
        
        if self.constraints:
            prompt_parts.append("\nConstraints:")
            for constraint in self.constraints:
                prompt_parts.append(f"- {constraint}")
        
        prompt_parts.append(f"\nExpected Output:\n{self.expected_output}")
        
        return "\n".join(prompt_parts)


class FoundryLocalClient:
    """Client for interacting with Microsoft Foundry Local."""
    
    def __init__(self, base_url: str = "http://localhost:8080", model: str = "qwen2.5-0.5b"):
        """Initialize Foundry Local client."""
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="not-needed"  # Foundry Local doesn't require API key
        )
    
    def check_connection(self) -> bool:
        """Check if Foundry Local is accessible."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=3)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def generate_completion(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate completion using Foundry Local."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert software development assistant helping create optimized prompts for AI-assisted coding."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating completion: {str(e)}"
    
    def generate_with_tools(self, prompt: str, tools: List[dict]) -> str:
        """Generate completion with function calling support."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error with tools: {str(e)}"


class PromptAssistant:
    """Main assistant for generating vibe coding prompts."""
    
    def __init__(self, foundry_client: FoundryLocalClient):
        """Initialize prompt assistant with Foundry Local client."""
        self.client = foundry_client
    
    def analyze_intent(self, user_request: str) -> dict:
        """Analyze user's request to extract intent and category."""
        analysis_prompt = f"""Analyze this software development request and extract:
1. Primary intent (what the user wants to accomplish)
2. Category (code_generation, refactoring, debugging, testing, documentation, architecture, data_pipeline, or ui_design)
3. Key requirements (list 3-5 main requirements)
4. Technical constraints (if any mentioned)

User request: "{user_request}"

Provide a structured analysis."""

        response = self.client.generate_completion(analysis_prompt, max_tokens=500)
        
        # Parse response (simplified - in production, use structured output)
        return {
            "intent": user_request,
            "category": "code_generation",  # Default fallback
            "requirements": [user_request],
            "constraints": []
        }
    
    def generate_prompt_template(self, analysis: dict) -> PromptTemplate:
        """Generate a structured prompt template from analysis."""
        template_prompt = f"""Create a comprehensive development prompt for:
Intent: {analysis['intent']}
Category: {analysis['category']}

Generate:
1. Clear context explanation
2. Detailed requirements list
3. Technical constraints
4. Expected output description

Format as a structured prompt suitable for AI code generation."""

        response = self.client.generate_completion(template_prompt, max_tokens=800)
        
        # Create template (simplified parsing)
        return PromptTemplate(
            category=analysis['category'],
            intent=analysis['intent'],
            context=f"Create a solution for: {analysis['intent']}",
            requirements=analysis.get('requirements', [analysis['intent']]),
            constraints=analysis.get('constraints', []),
            expected_output="Functional, well-documented code with tests"
        )
    
    def refine_prompt(self, template: PromptTemplate, feedback: str) -> PromptTemplate:
        """Refine prompt based on user feedback."""
        refinement_prompt = f"""Improve this development prompt based on feedback:

Current prompt:
{template.to_prompt()}

User feedback: {feedback}

Provide an enhanced version addressing the feedback."""

        response = self.client.generate_completion(refinement_prompt, max_tokens=800)
        
        # For simplicity, return modified template
        # In production, parse the response to update specific fields
        return template
    
    def generate_vibe_coding_prompt(self, user_request: str) -> str:
        """Generate complete vibe coding prompt from user request."""
        print("🔍 Analyzing your request with Foundry Local...")
        analysis = self.analyze_intent(user_request)
        
        print(f"✓ Detected category: {analysis['category']}")
        print(f"✓ Intent: {analysis['intent'][:60]}...")
        
        print("📝 Generating optimized vibe coding prompt...")
        template = self.generate_prompt_template(analysis)
        
        return template.to_prompt()
    
    def interactive_session(self):
        """Run interactive prompt generation session."""
        print("=" * 70)
        print("🤖 Prompt Assistant for Vibe Coding (Powered by Foundry Local)")
        print("=" * 70)
        print("Describe what you want to build, and I'll create an optimized prompt")
        print("for AI-assisted coding. Type 'exit' to quit or 'refine' to improve.")
        print("=" * 70)
        print()
        
        current_prompt = None
        current_template = None
        
        while True:
            user_input = input("💬 Your request: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'refine' and current_template:
                feedback = input("💬 What would you like to improve? ")
                print("🔄 Refining prompt...")
                current_template = self.refine_prompt(current_template, feedback)
                current_prompt = current_template.to_prompt()
                print("\n✨ REFINED VIBE CODING PROMPT:\n")
                print("-" * 70)
                print(current_prompt)
                print("-" * 70)
                print()
                continue
            
            # Generate new prompt
            try:
                analysis = self.analyze_intent(user_input)
                print(f"✓ Category: {analysis['category']}")
                
                current_template = self.generate_prompt_template(analysis)
                current_prompt = current_template.to_prompt()
                
                print("\n✨ GENERATED VIBE CODING PROMPT:\n")
                print("-" * 70)
                print(current_prompt)
                print("-" * 70)
                print("\n💡 Copy this prompt and use it with GitHub Copilot, Cursor, or VS Code!")
                print("   Type 'refine' to improve or enter a new request.\n")
            
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("Please try again with a different request.\n")


def check_foundry_local_status():
    """Check if Foundry Local is running and accessible."""
    print("🔍 Checking Foundry Local connection...")
    
    try:
        response = requests.get("http://localhost:8080/health", timeout=3)
        if response.status_code == 200:
            print("✓ Foundry Local is running")
            return True
    except requests.RequestException:
        pass
    
    print("⚠️  Warning: Foundry Local not detected at http://localhost:8080")
    print("   The assistant will have limited functionality.")
    print("   Install Foundry Local: See FOUNDRY_LOCAL_SETUP.md")
    print()
    return False


def main():
    """Main entry point for prompt assistant."""
    # Check environment
    model = os.environ.get("FOUNDRY_MODEL", "qwen2.5-0.5b")
    
    # Verify Foundry Local
    foundry_available = check_foundry_local_status()
    
    # Initialize client
    client = FoundryLocalClient(model=model)
    assistant = PromptAssistant(client)
    
    # Check for CLI mode (single request)
    if len(sys.argv) > 1:
        user_request = " ".join(sys.argv[1:])
        print(f"💬 Request: {user_request}\n")
        
        try:
            prompt = assistant.generate_vibe_coding_prompt(user_request)
            print("\n✨ GENERATED VIBE CODING PROMPT:\n")
            print("-" * 70)
            print(prompt)
            print("-" * 70)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            sys.exit(1)
    else:
        # Interactive mode
        assistant.interactive_session()


if __name__ == "__main__":
    main()
