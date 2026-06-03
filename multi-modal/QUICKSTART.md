# Quick Start Guide

This guide will help you get started with the Multi-Modal AI project.

## Step 1: Verify Prerequisites

```bash
# Check Python version (3.9+ required)
python3 --version

# Check if virtual environment is activated
which python  # Should show path to venv/bin/python
```

## Step 2: Activate Virtual Environment

If not already activated:

```bash
cd /home/laptop/EXERCISES/AUTONOMOUS/autonomous-artificial-intelligence/multi-modal
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

## Step 3: Install and Start Ollama

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server (in a separate terminal)
ollama serve

# Download lightweight models (in another terminal, with venv activated)
ollama pull llama3.2:3b      # Text model (2GB)
ollama pull llava:7b          # Vision model (4.5GB)

# Verify models are downloaded
ollama list
```

## Step 4: Test Text Model

```bash
# Activate virtual environment
source venv/bin/activate

# Test with a single prompt
python scripts/text_model.py --prompt "Explain machine learning in simple terms"

# Or run in interactive mode
python scripts/text_model.py --interactive
```

## Step 5: Test Image Model

```bash
# First, download a sample image or use your own
# Example: download a test image
wget https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/400px-Cat03.jpg -O test_image.jpg

# Analyze the image
python scripts/image_model.py --image test_image.jpg --prompt "Describe this image in detail"

# Extract text from an image (OCR)
python scripts/image_model.py --image test_image.jpg --ocr
```

## Step 6: Test Voice Model

```bash
# Note: Voice model requires audio file
# If you have a WAV file:
python scripts/voice_model.py --audio your_audio.wav

# Analyze audio properties only
python scripts/voice_model.py --audio your_audio.wav --analyze-only
```

## Step 7: Test AI Agent

```bash
# Interactive mode (recommended for first try)
python scripts/ai_agent.py --mode interactive

# Within interactive mode, you can:
# - Ask questions: "What is quantum computing?"
# - Analyze images: image /path/to/image.jpg
# - Clear history: clear
# - Exit: exit

# Or run demo scenarios
python scripts/ai_agent.py --demo
```

## Step 8: Start FastAPI Server

```bash
# Start the REST API server
python scripts/api_server.py

# In another terminal, test the API:

# Health check
curl http://localhost:8000/health

# Text processing
curl -X POST http://localhost:8000/text \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain neural networks"}'

# Image processing
curl -X POST http://localhost:8000/image \
  -F "image=@test_image.jpg" \
  -F "prompt=What is in this image?"

# View API documentation
# Open browser: http://localhost:8000/docs
```

## Step 9: Test with Open WebUI (Optional)

```bash
# Start Open WebUI with Docker
docker run -d \
  --name open-webui \
  -p 3000:8080 \
  -v open-webui:/app/backend/data \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main

# Access in browser
# http://localhost:3000

# Select model from dropdown (llava:7b for multi-modal)
# Click + icon to upload images
# Type questions about the image
```

## Common Issues and Solutions

### Issue: "Cannot connect to Ollama server"

**Solution:**
```bash
# Check if Ollama is running
ps aux | grep ollama

# If not running, start it
ollama serve

# Verify it's accessible
curl http://localhost:11434/api/tags
```

### Issue: "Module not found" errors

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Model not found" errors

**Solution:**
```bash
# List available models
ollama list

# Download missing model
ollama pull llama3.2:3b
ollama pull llava:7b
```

### Issue: Out of memory when loading models

**Solution:**
```bash
# Use smaller models
ollama pull llama3.2:1b       # Smaller text model (0.9GB)
ollama pull bakllava:latest   # Alternative vision model

# Update scripts to use smaller model
python scripts/text_model.py --model llama3.2:1b
```

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Explore multi-modal capabilities by combining text and images
3. Build custom agents using the AI agent framework
4. Integrate the FastAPI server into your applications
5. Experiment with different models and prompts

## Useful Commands Reference

```bash
# Virtual environment
source venv/bin/activate      # Activate
deactivate                    # Deactivate

# Ollama
ollama serve                  # Start server
ollama list                   # List models
ollama pull <model>           # Download model
ollama rm <model>             # Remove model

# Scripts (all require activated venv)
python scripts/text_model.py --help
python scripts/image_model.py --help
python scripts/voice_model.py --help
python scripts/ai_agent.py --help
python scripts/api_server.py
```

## Getting Help

- Check logs: Look at terminal output for error messages
- Verify installation: Ensure all dependencies are installed
- Test connectivity: Ensure Ollama server is running and accessible
- Check models: Verify models are downloaded with `ollama list`
- Review documentation: Read README.md for detailed information

Happy exploring multi-modal AI!
