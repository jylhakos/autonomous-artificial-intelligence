#!/usr/bin/env python3
"""
Text Model Script - Process text queries using Ollama local server

This script demonstrates text-only interaction with lightweight language models
deployed on a local Ollama server. It supports both single queries and
interactive conversation mode.

Usage:
    python text_model.py --prompt "Your question here"
    python text_model.py --interactive
"""

import argparse
import sys
import json
from typing import Optional, Dict, Any
import requests


class TextModelClient:
    """Client for interacting with Ollama text models"""
    
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        """
        Initialize the text model client
        
        Args:
            model: Name of the Ollama model to use
            base_url: Base URL of the Ollama server
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.conversation_history = []
        
    def check_connection(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate(self, prompt: str, stream: bool = True, context: Optional[list] = None) -> Dict[str, Any]:
        """
        Generate text response from the model
        
        Args:
            prompt: Input text prompt
            stream: Whether to stream the response
            context: Conversation context for continuity
            
        Returns:
            Dictionary containing response and metadata
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }
        
        if context:
            payload["context"] = context
        
        try:
            response = requests.post(self.api_url, json=payload, stream=stream, timeout=120)
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json()
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def _handle_stream_response(self, response) -> Dict[str, Any]:
        """Handle streaming response from Ollama"""
        full_response = ""
        context = None
        
        print("\nResponse: ", end="", flush=True)
        
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line.decode('utf-8'))
                    token = json_response.get("response", "")
                    full_response += token
                    print(token, end="", flush=True)
                    
                    if json_response.get("done", False):
                        context = json_response.get("context")
                        print("\n")
                        
                except json.JSONDecodeError:
                    continue
        
        return {
            "response": full_response,
            "context": context
        }
    
    def chat(self, user_message: str) -> str:
        """
        Chat with the model maintaining conversation context
        
        Args:
            user_message: User's message
            
        Returns:
            Model's response
        """
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Build context from conversation history
        context_prompt = self._build_context_prompt()
        
        result = self.generate(context_prompt, stream=True)
        
        if "error" in result:
            return result["error"]
        
        response_text = result.get("response", "")
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        return response_text
    
    def _build_context_prompt(self) -> str:
        """Build prompt with conversation context"""
        prompt = ""
        for message in self.conversation_history[-6:]:  # Keep last 6 messages
            role = message["role"].capitalize()
            content = message["content"]
            prompt += f"{role}: {content}\n"
        
        return prompt + "Assistant:"
    
    def interactive_mode(self):
        """Run interactive conversation mode"""
        print(f"\n{'='*60}")
        print(f"Interactive Text Model Chat (Model: {self.model})")
        print(f"{'='*60}")
        print("Type 'exit', 'quit', or 'q' to end the conversation")
        print("Type 'clear' to reset conversation history\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("\nConversation history cleared.")
                    continue
                
                self.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Text Model Client for Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="Ollama model to use (default: llama3.2:3b)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single text prompt to process"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive conversation mode"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server base URL (default: http://localhost:11434)"
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = TextModelClient(model=args.model, base_url=args.base_url)
    
    # Check connection
    print(f"Connecting to Ollama server at {args.base_url}...")
    if not client.check_connection():
        print(f"Error: Cannot connect to Ollama server at {args.base_url}")
        print("Please ensure Ollama is running: ollama serve")
        sys.exit(1)
    
    print(f"Connected successfully! Using model: {args.model}\n")
    
    # Process based on mode
    if args.interactive:
        client.interactive_mode()
    elif args.prompt:
        print(f"Prompt: {args.prompt}")
        result = client.generate(args.prompt, stream=True)
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
    else:
        print("Error: Please specify either --prompt or --interactive mode")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
