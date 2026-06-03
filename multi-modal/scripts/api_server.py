#!/usr/bin/env python3
"""
FastAPI Server for Multi-Modal AI Services

This FastAPI server provides REST API endpoints for accessing multi-modal AI
capabilities, including text processing, image analysis, and audio transcription.

The server acts as a bridge between clients and the local Ollama server,
providing a unified interface for multi-modal model interactions.

Usage:
    python api_server.py
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import base64
import os
import tempfile
import json
import requests
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Modal AI API",
    description="REST API for text, image, and audio processing with local Ollama models",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEXT_MODEL = os.getenv("TEXT_MODEL", "llama3.2:3b")
VISION_MODEL = os.getenv("VISION_MODEL", "llava:7b")


# Request/Response Models
class TextRequest(BaseModel):
    """Request model for text processing"""
    prompt: str = Field(..., description="Text prompt to process")
    model: Optional[str] = Field(TEXT_MODEL, description="Model to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(2000, ge=1, le=4096, description="Maximum tokens to generate")


class TextResponse(BaseModel):
    """Response model for text processing"""
    response: str
    model: str
    timestamp: str


class ImageAnalysisResponse(BaseModel):
    """Response model for image analysis"""
    response: str
    model: str
    image_info: Dict[str, Any]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    ollama_status: str
    available_models: List[str]
    timestamp: str


# Helper Functions
def check_ollama_connection() -> bool:
    """Check if Ollama server is accessible"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_available_models() -> List[str]:
    """Get list of available models from Ollama"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        return []
    except requests.exceptions.RequestException:
        return []


def call_ollama_generate(payload: Dict[str, Any]) -> requests.Response:
    """Call Ollama generate API"""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    return requests.post(url, json=payload, stream=True, timeout=180)


def stream_ollama_response(response: requests.Response):
    """Stream response from Ollama"""
    for line in response.iter_lines():
        if line:
            try:
                json_data = json.loads(line.decode('utf-8'))
                yield json.dumps(json_data) + "\n"
            except json.JSONDecodeError:
                continue


def collect_full_response(response: requests.Response) -> str:
    """Collect full response from streaming Ollama API"""
    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_data = json.loads(line.decode('utf-8'))
                full_response += json_data.get("response", "")
                if json_data.get("done", False):
                    break
            except json.JSONDecodeError:
                continue
    return full_response


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Modal AI API Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    ollama_connected = check_ollama_connection()
    available_models = get_available_models() if ollama_connected else []
    
    return HealthResponse(
        status="healthy" if ollama_connected else "degraded",
        ollama_status="connected" if ollama_connected else "disconnected",
        available_models=available_models,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/text", response_model=TextResponse)
async def process_text(request: TextRequest):
    """
    Process text-only request
    
    This endpoint processes text prompts using the specified language model.
    """
    if not check_ollama_connection():
        raise HTTPException(status_code=503, detail="Ollama server not available")
    
    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": False,
        "options": {
            "temperature": request.temperature,
            "num_predict": request.max_tokens
        }
    }
    
    try:
        response = call_ollama_generate(payload)
        response.raise_for_status()
        
        # For non-streaming, we get a single JSON response
        full_response = collect_full_response(response)
        
        return TextResponse(
            response=full_response,
            model=request.model,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {str(e)}")


@app.post("/text/stream")
async def process_text_stream(request: TextRequest):
    """
    Process text request with streaming response
    
    Returns a streaming response for real-time token generation.
    """
    if not check_ollama_connection():
        raise HTTPException(status_code=503, detail="Ollama server not available")
    
    payload = {
        "model": request.model,
        "prompt": request.prompt,
        "stream": True,
        "options": {
            "temperature": request.temperature,
            "num_predict": request.max_tokens
        }
    }
    
    try:
        response = call_ollama_generate(payload)
        response.raise_for_status()
        
        return StreamingResponse(
            stream_ollama_response(response),
            media_type="application/x-ndjson"
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {str(e)}")


@app.post("/image", response_model=ImageAnalysisResponse)
async def process_image(
    image: UploadFile = File(..., description="Image file to analyze"),
    prompt: str = Form(..., description="Question or instruction about the image"),
    model: str = Form(VISION_MODEL, description="Vision model to use")
):
    """
    Process image with text prompt
    
    This endpoint analyzes images and answers questions about them using
    vision-capable language models.
    """
    if not check_ollama_connection():
        raise HTTPException(status_code=503, detail="Ollama server not available")
    
    # Validate image file
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"]
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image type. Allowed types: {', '.join(allowed_types)}"
        )
    
    # Read and encode image
    try:
        image_data = await image.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Get image info
        image_info = {
            "filename": image.filename,
            "content_type": image.content_type,
            "size_bytes": len(image_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
    
    # Call Ollama with image
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False
    }
    
    try:
        response = call_ollama_generate(payload)
        response.raise_for_status()
        
        full_response = collect_full_response(response)
        
        return ImageAnalysisResponse(
            response=full_response,
            model=model,
            image_info=image_info,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {str(e)}")


@app.post("/audio")
async def process_audio(
    audio: UploadFile = File(..., description="Audio file to process"),
    task: str = Form("transcribe", description="Task: transcribe or analyze")
):
    """
    Process audio file
    
    Note: This is a placeholder endpoint. Full audio processing requires
    additional libraries like Whisper for transcription.
    """
    if not check_ollama_connection():
        raise HTTPException(status_code=503, detail="Ollama server not available")
    
    # Validate audio file
    allowed_types = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg"]
    if audio.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio type. Allowed types: {', '.join(allowed_types)}"
        )
    
    # Save audio temporarily
    try:
        audio_data = await audio.read()
        
        # This is a placeholder - real implementation would use Whisper or similar
        return JSONResponse({
            "message": "Audio processing placeholder",
            "note": "Integrate Whisper or similar for real transcription",
            "audio_info": {
                "filename": audio.filename,
                "content_type": audio.content_type,
                "size_bytes": len(audio_data)
            },
            "task": task,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {str(e)}")


@app.get("/models")
async def list_models():
    """List available models from Ollama"""
    if not check_ollama_connection():
        raise HTTPException(status_code=503, detail="Ollama server not available")
    
    models = get_available_models()
    
    return {
        "models": models,
        "count": len(models),
        "timestamp": datetime.utcnow().isoformat()
    }


# Error Handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "path": str(request.url)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )


# Run server
if __name__ == "__main__":
    import uvicorn
    
    print("Starting Multi-Modal AI API Server...")
    print(f"Ollama Base URL: {OLLAMA_BASE_URL}")
    print(f"Text Model: {TEXT_MODEL}")
    print(f"Vision Model: {VISION_MODEL}")
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
