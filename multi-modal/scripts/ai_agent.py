#!/usr/bin/env python3
"""
AI Agent with Multi-Modal Capabilities

This script demonstrates an intelligent AI agent that can process text, images, and 
audio using Langchain/Langgraph for orchestration and Ollama for model inference.

The agent can:
- Maintain conversation context
- Process multi-modal inputs (text + images)
- Make decisions based on user requests
- Use tools and external functions

Usage:
    python ai_agent.py --mode text
    python ai_agent.py --mode multimodal --image sample.jpg
    python ai_agent.py --mode interactive
"""

import argparse
import sys
import os
import base64
from typing import Optional, Dict, Any, List
import json

try:
    from langchain.llms import Ollama
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferMemory
    from langchain.agents import Tool, AgentExecutor, create_react_agent
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: Langchain not installed. Install with: pip install langchain langchain-community")

import requests


class MultiModalAgent:
    """
    AI Agent with multi-modal processing capabilities
    
    This agent uses Langchain for orchestration and can process:
    - Text queries and conversations
    - Images with vision models
    - Tool usage and function calling
    """
    
    def __init__(
        self,
        text_model: str = "llama3.2:3b",
        vision_model: str = "llava:7b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the multi-modal agent
        
        Args:
            text_model: Ollama text model name
            vision_model: Ollama vision model name
            base_url: Ollama server base URL
        """
        self.text_model = text_model
        self.vision_model = vision_model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.conversation_history = []
        
        # Initialize Langchain components if available
        if LANGCHAIN_AVAILABLE:
            self.llm = Ollama(model=text_model, base_url=base_url)
            self.memory = ConversationBufferMemory(memory_key="chat_history")
        else:
            self.llm = None
            self.memory = None
    
    def check_connection(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def process_text(self, prompt: str) -> str:
        """
        Process text-only query
        
        Args:
            prompt: User's text input
            
        Returns:
            Agent's response
        """
        print("\n" + "="*60)
        print("AGENT PROCESSING (Text Mode)")
        print("="*60)
        
        if LANGCHAIN_AVAILABLE and self.llm:
            # Use Langchain for structured processing
            response = self.llm.invoke(prompt)
            return response
        else:
            # Fallback to direct API call
            return self._direct_api_call(prompt, self.text_model)
    
    def process_multimodal(self, text_prompt: str, image_path: str) -> str:
        """
        Process multi-modal input (text + image)
        
        Args:
            text_prompt: User's text query
            image_path: Path to image file
            
        Returns:
            Agent's response
        """
        print("\n" + "="*60)
        print("AGENT PROCESSING (Multi-Modal Mode)")
        print("="*60)
        print(f"Image: {image_path}")
        print(f"Query: {text_prompt}\n")
        
        # Encode image
        image_base64 = self.encode_image(image_path)
        if not image_base64:
            return "Error: Failed to load image"
        
        # Call vision model
        payload = {
            "model": self.vision_model,
            "prompt": text_prompt,
            "images": [image_base64],
            "stream": True
        }
        
        return self._stream_response(payload)
    
    def _direct_api_call(self, prompt: str, model: str) -> str:
        """Make direct API call to Ollama"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True
        }
        
        return self._stream_response(payload)
    
    def _stream_response(self, payload: Dict) -> str:
        """Handle streaming response"""
        try:
            response = requests.post(self.api_url, json=payload, stream=True, timeout=180)
            response.raise_for_status()
            
            full_response = ""
            print("Response: ", end="", flush=True)
            
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line.decode('utf-8'))
                        token = json_response.get("response", "")
                        full_response += token
                        print(token, end="", flush=True)
                        
                        if json_response.get("done", False):
                            print("\n")
                            
                    except json.JSONDecodeError:
                        continue
            
            return full_response
            
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"
    
    def interactive_mode(self):
        """Run interactive conversation mode"""
        print("\n" + "="*60)
        print("AI AGENT - Interactive Mode")
        print("="*60)
        print(f"Text Model: {self.text_model}")
        print(f"Vision Model: {self.vision_model}")
        print("\nCommands:")
        print("  'exit' or 'quit' - End conversation")
        print("  'clear' - Clear conversation history")
        print("  'image <path>' - Analyze an image")
        print("  'help' - Show this help message")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nAgent: Goodbye! It was nice working with you.")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    if self.memory:
                        self.memory.clear()
                    print("\nAgent: Conversation history cleared.")
                    continue
                
                if user_input.lower() == 'help':
                    print("\nAgent: I'm a multi-modal AI agent. I can:")
                    print("  - Answer questions and have conversations")
                    print("  - Analyze images (use 'image <path>')")
                    print("  - Help with various tasks")
                    continue
                
                # Check if user wants to analyze an image
                if user_input.lower().startswith('image '):
                    parts = user_input.split(' ', 1)
                    if len(parts) > 1:
                        image_path = parts[1].strip()
                        if os.path.exists(image_path):
                            prompt = input("What would you like to know about this image? ")
                            response = self.process_multimodal(prompt, image_path)
                        else:
                            print(f"\nAgent: Sorry, I couldn't find the image at: {image_path}")
                            continue
                    else:
                        print("\nAgent: Please provide an image path after 'image'")
                        continue
                else:
                    # Regular text processing
                    response = self.process_text(user_input)
                
                # Store in conversation history
                self.conversation_history.append({
                    "user": user_input,
                    "agent": response
                })
                
            except KeyboardInterrupt:
                print("\n\nAgent: Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    def run_scenario_demo(self):
        """
        Run a demonstration scenario showing agent capabilities
        """
        print("\n" + "="*60)
        print("AI AGENT - Scenario Demonstration")
        print("="*60)
        print("\nScenario: Document Analysis Assistant")
        print("\nThe agent will demonstrate multi-modal capabilities by:")
        print("1. Understanding your request")
        print("2. Processing documents and images")
        print("3. Providing comprehensive analysis")
        print("="*60 + "\n")
        
        scenarios = [
            {
                "title": "Text Analysis",
                "prompt": "Explain the concept of neural networks in simple terms suitable for a beginner."
            },
            {
                "title": "Technical Question",
                "prompt": "What are the key differences between supervised and unsupervised learning?"
            },
            {
                "title": "Problem Solving",
                "prompt": "A user wants to deploy a machine learning model. What steps should they follow?"
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{'='*60}")
            print(f"Scenario {i}: {scenario['title']}")
            print(f"{'='*60}")
            print(f"\nUser Query: {scenario['prompt']}\n")
            
            response = self.process_text(scenario['prompt'])
            
            input("\nPress Enter to continue to next scenario...")
        
        print("\n" + "="*60)
        print("Demonstration Complete!")
        print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Modal AI Agent with Langchain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ai_agent.py --mode text --prompt "Explain quantum computing"
  python ai_agent.py --mode multimodal --image photo.jpg --prompt "What's in this image?"
  python ai_agent.py --mode interactive
  python ai_agent.py --demo
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["text", "multimodal", "interactive"],
        default="interactive",
        help="Agent operation mode"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for the agent"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file (for multimodal mode)"
    )
    
    parser.add_argument(
        "--text-model",
        type=str,
        default="llama3.2:3b",
        help="Ollama text model (default: llama3.2:3b)"
    )
    
    parser.add_argument(
        "--vision-model",
        type=str,
        default="llava:7b",
        help="Ollama vision model (default: llava:7b)"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server base URL"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration scenario"
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = MultiModalAgent(
        text_model=args.text_model,
        vision_model=args.vision_model,
        base_url=args.base_url
    )
    
    # Check connection
    print(f"Connecting to Ollama server at {args.base_url}...")
    if not agent.check_connection():
        print(f"Error: Cannot connect to Ollama server")
        print("Please ensure Ollama is running: ollama serve")
        sys.exit(1)
    
    print(f"Connected successfully!")
    
    # Run based on mode
    if args.demo:
        agent.run_scenario_demo()
    elif args.mode == "interactive":
        agent.interactive_mode()
    elif args.mode == "text":
        if not args.prompt:
            print("Error: --prompt required for text mode")
            sys.exit(1)
        response = agent.process_text(args.prompt)
    elif args.mode == "multimodal":
        if not args.image or not args.prompt:
            print("Error: --image and --prompt required for multimodal mode")
            sys.exit(1)
        response = agent.process_multimodal(args.prompt, args.image)


if __name__ == "__main__":
    main()
