# Document Analysis Agent

This sample demonstrates a privacy-focused document analysis AI agent running entirely locally using Docker and Ollama.

## Features

- PDF document loading and parsing
- Local vector storage with ChromaDB
- Document question answering
- Summarization and analysis
- No data leaves your infrastructure
- GPU acceleration support

## Prerequisites

- Docker and Docker Compose installed
- At least 16GB RAM
- 20GB free disk space
- Optional: NVIDIA GPU for faster inference

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Pull the Ollama model:
```bash
docker pull ollama/ollama
docker run -d --name ollama ollama/ollama
docker exec ollama ollama pull qwen2.5:32b
```

3. Set environment variables (optional):
```bash
export OLLAMA_HOST=http://localhost:11434
export CHROMADB_PATH=./data/chromadb
```

## Running with Docker Compose

```bash
docker-compose up
```

This starts:
- Ollama model server on port 11434
- Document analysis API on port 8000

## Running Locally

```bash
python main.py
```

## API Usage

```bash
# Upload a document
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "document_id": "doc-123"}'

# Generate summary
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"document_id": "doc-123"}'
```

## GPU Support

To enable GPU acceleration, modify docker-compose.yml:

```yaml
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Documentation

See the main deployment guide: [Docker-Ollama.md](../../Docker-Ollama.md)
