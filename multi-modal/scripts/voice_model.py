#!/usr/bin/env python3
"""
Voice Model Script - Process audio input using speech recognition and Ollama

This script demonstrates audio processing capabilities including:
- Speech-to-text transcription
- Audio analysis
- Voice command processing with language models

Usage:
    python voice_model.py --audio path/to/audio.wav
    python voice_model.py --record --duration 5
"""

import argparse
import sys
import os
from typing import Optional, Dict, Any
import requests
import json

try:
    import soundfile as sf
    import numpy as np
    AUDIO_SUPPORT = True
except ImportError:
    AUDIO_SUPPORT = False


class VoiceModelClient:
    """Client for processing audio with speech recognition and LLM"""
    
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        """
        Initialize the voice model client
        
        Args:
            model: Name of the Ollama model for text processing
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
    
    def load_audio(self, audio_path: str) -> Optional[tuple]:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate) or None if error
        """
        if not AUDIO_SUPPORT:
            print("Error: soundfile library not installed")
            print("Install it with: pip install soundfile")
            return None
        
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            return None
        
        try:
            audio_data, sample_rate = sf.read(audio_path)
            print(f"\nAudio loaded successfully:")
            print(f"  Duration: {len(audio_data) / sample_rate:.2f} seconds")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Channels: {audio_data.ndim}")
            return audio_data, sample_rate
        except Exception as e:
            print(f"Error loading audio: {str(e)}")
            return None
    
    def transcribe_audio_simulation(self, audio_path: str) -> str:
        """
        Simulate audio transcription
        
        Note: This is a placeholder. For real transcription, you would use:
        - Whisper (OpenAI): pip install openai-whisper
        - SpeechRecognition: pip install SpeechRecognition
        - Faster-Whisper: pip install faster-whisper
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        print("\n" + "="*60)
        print("AUDIO TRANSCRIPTION (Simulation)")
        print("="*60)
        print("\nNote: This is a simulated transcription.")
        print("For real transcription, integrate one of these:")
        print("  - Whisper: pip install openai-whisper")
        print("  - Faster-Whisper: pip install faster-whisper")
        print("  - Google Speech API")
        print("\nSimulated transcription process:")
        
        # Load and analyze audio
        audio_info = self.load_audio(audio_path)
        if not audio_info:
            return ""
        
        # Simulate transcription
        simulated_text = (
            "Hello, this is a simulated transcription of the audio file. "
            "In a production environment, this would be the actual speech-to-text "
            "output from a model like Whisper or Google Speech API."
        )
        
        print(f"\nTranscribed text: {simulated_text}")
        return simulated_text
    
    def process_voice_command(self, transcribed_text: str) -> Dict[str, Any]:
        """
        Process transcribed text with language model
        
        Args:
            transcribed_text: Text from speech recognition
            
        Returns:
            Model response dictionary
        """
        if not transcribed_text:
            return {"error": "No transcribed text provided"}
        
        print("\n" + "="*60)
        print("PROCESSING WITH LANGUAGE MODEL")
        print("="*60)
        
        prompt = f"""You are a voice assistant. A user said the following:

"{transcribed_text}"

Provide a helpful and natural response."""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        
        try:
            response = requests.post(self.api_url, json=payload, stream=True, timeout=120)
            response.raise_for_status()
            
            full_response = ""
            print("\nAssistant response: ", end="", flush=True)
            
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
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def analyze_audio_properties(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze audio file properties
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of audio properties
        """
        audio_info = self.load_audio(audio_path)
        if not audio_info:
            return {}
        
        audio_data, sample_rate = audio_info
        
        # Calculate basic properties
        duration = len(audio_data) / sample_rate
        
        if audio_data.ndim == 1:
            # Mono
            rms_level = np.sqrt(np.mean(audio_data**2))
            peak_level = np.max(np.abs(audio_data))
        else:
            # Stereo or multi-channel
            rms_level = np.sqrt(np.mean(audio_data**2, axis=0))
            peak_level = np.max(np.abs(audio_data), axis=0)
        
        properties = {
            "duration": duration,
            "sample_rate": sample_rate,
            "channels": 1 if audio_data.ndim == 1 else audio_data.shape[1],
            "rms_level": float(rms_level) if isinstance(rms_level, (int, float, np.number)) else rms_level.tolist(),
            "peak_level": float(peak_level) if isinstance(peak_level, (int, float, np.number)) else peak_level.tolist()
        }
        
        print("\n" + "="*60)
        print("AUDIO ANALYSIS")
        print("="*60)
        print(f"Duration: {properties['duration']:.2f} seconds")
        print(f"Sample Rate: {properties['sample_rate']} Hz")
        print(f"Channels: {properties['channels']}")
        print(f"RMS Level: {properties['rms_level']}")
        print(f"Peak Level: {properties['peak_level']}")
        
        return properties


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Voice Model Client - Audio processing with Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file (WAV, MP3, etc.)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="Ollama model to use (default: llama3.2:3b)"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server base URL"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze audio properties without transcription"
    )
    
    args = parser.parse_args()
    
    if not args.audio:
        print("Error: Please provide an audio file with --audio")
        parser.print_help()
        sys.exit(1)
    
    # Initialize client
    client = VoiceModelClient(model=args.model, base_url=args.base_url)
    
    # Check connection
    print(f"Connecting to Ollama server at {args.base_url}...")
    if not client.check_connection():
        print(f"Error: Cannot connect to Ollama server")
        print("Please ensure Ollama is running: ollama serve")
        sys.exit(1)
    
    print(f"Connected successfully! Using model: {args.model}\n")
    
    # Process audio
    if args.analyze_only:
        client.analyze_audio_properties(args.audio)
    else:
        # Full pipeline: analyze -> transcribe -> process
        client.analyze_audio_properties(args.audio)
        transcribed_text = client.transcribe_audio_simulation(args.audio)
        
        if transcribed_text:
            result = client.process_voice_command(transcribed_text)
            if "error" in result:
                print(f"\nError: {result['error']}")
                sys.exit(1)


if __name__ == "__main__":
    main()
