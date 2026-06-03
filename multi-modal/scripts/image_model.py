#!/usr/bin/env python3
"""
Image Model Script - Process images with vision-capable language models

This script demonstrates multi-modal image processing using Ollama's vision models.
It supports image description, visual question answering, and OCR tasks.

Usage:
    python image_model.py --image path/to/image.jpg --prompt "Describe this image"
    python image_model.py --image photo.png --prompt "What text do you see?"
"""

import argparse
import sys
import os
import base64
from typing import Optional, Dict, Any
import requests
import json

try:
    from PIL import Image
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False


class ImageModelClient:
    """Client for processing images with Ollama vision models"""
    
    def __init__(self, model: str = "llava:7b", base_url: str = "http://localhost:11434"):
        """
        Initialize the image model client
        
        Args:
            model: Name of the Ollama vision model (must support images)
            base_url: Base URL of the Ollama server
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def check_connection(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load and validate image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object or None if error
        """
        if not IMAGE_SUPPORT:
            print("Error: PIL library not installed")
            print("Install it with: pip install Pillow")
            return None
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return None
        
        try:
            image = Image.open(image_path)
            print(f"\nImage loaded successfully:")
            print(f"  Format: {image.format}")
            print(f"  Size: {image.size[0]}x{image.size[1]} pixels")
            print(f"  Mode: {image.mode}")
            return image
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return None
    
    def encode_image_base64(self, image_path: str) -> Optional[str]:
        """
        Encode image to base64 string for API transmission
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string or None if error
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            return None
    
    def analyze_image(self, image_path: str, prompt: str, stream: bool = True) -> Dict[str, Any]:
        """
        Analyze image with vision model
        
        Args:
            image_path: Path to image file
            prompt: Text prompt/question about the image
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing response and metadata
        """
        # Load and validate image
        image = self.load_image(image_path)
        if not image:
            return {"error": "Failed to load image"}
        
        # Encode image to base64
        image_base64 = self.encode_image_base64(image_path)
        if not image_base64:
            return {"error": "Failed to encode image"}
        
        print("\n" + "="*60)
        print("PROCESSING IMAGE WITH VISION MODEL")
        print("="*60)
        print(f"Prompt: {prompt}\n")
        
        # Prepare API payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": stream
        }
        
        try:
            response = requests.post(self.api_url, json=payload, stream=stream, timeout=180)
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
        
        return {"response": full_response}
    
    def extract_text_ocr(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from image using OCR capabilities
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing extracted text
        """
        prompt = """Please extract all text visible in this image. 
List the text exactly as it appears, maintaining the original formatting where possible.
If there is no text in the image, respond with 'No text detected.'"""
        
        return self.analyze_image(image_path, prompt)
    
    def describe_image(self, image_path: str, detail_level: str = "detailed") -> Dict[str, Any]:
        """
        Generate detailed description of image
        
        Args:
            image_path: Path to image file
            detail_level: Level of detail (brief, detailed, comprehensive)
            
        Returns:
            Dictionary containing description
        """
        prompts = {
            "brief": "Provide a brief one-sentence description of this image.",
            "detailed": "Describe this image in detail, including objects, people, setting, colors, and atmosphere.",
            "comprehensive": """Provide a comprehensive analysis of this image including:
1. Main subjects and objects
2. Setting and environment
3. Colors and lighting
4. Composition and perspective
5. Mood or atmosphere
6. Any text visible
7. Notable details"""
        }
        
        prompt = prompts.get(detail_level, prompts["detailed"])
        return self.analyze_image(image_path, prompt)
    
    def answer_visual_question(self, image_path: str, question: str) -> Dict[str, Any]:
        """
        Answer specific question about the image
        
        Args:
            image_path: Path to image file
            question: Question about the image
            
        Returns:
            Dictionary containing answer
        """
        prompt = f"Based on this image, please answer the following question: {question}"
        return self.analyze_image(image_path, prompt)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Image Model Client - Vision analysis with Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_model.py --image photo.jpg --prompt "Describe this image"
  python image_model.py --image document.png --ocr
  python image_model.py --image scene.jpg --describe detailed
        """
    )
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image file"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt/question about the image"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="llava:7b",
        help="Ollama vision model to use (default: llava:7b)"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server base URL"
    )
    
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Extract text from image (OCR)"
    )
    
    parser.add_argument(
        "--describe",
        type=str,
        choices=["brief", "detailed", "comprehensive"],
        help="Generate image description with specified detail level"
    )
    
    args = parser.parse_args()
    
    # Initialize client
    client = ImageModelClient(model=args.model, base_url=args.base_url)
    
    # Check connection
    print(f"Connecting to Ollama server at {args.base_url}...")
    if not client.check_connection():
        print(f"Error: Cannot connect to Ollama server")
        print("Please ensure Ollama is running: ollama serve")
        sys.exit(1)
    
    print(f"Connected successfully! Using model: {args.model}")
    
    # Process based on mode
    result = None
    
    if args.ocr:
        result = client.extract_text_ocr(args.image)
    elif args.describe:
        result = client.describe_image(args.image, args.describe)
    elif args.prompt:
        result = client.analyze_image(args.image, args.prompt)
    else:
        print("\nError: Please specify one of: --prompt, --ocr, or --describe")
        parser.print_help()
        sys.exit(1)
    
    # Check for errors
    if result and "error" in result:
        print(f"\nError: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
