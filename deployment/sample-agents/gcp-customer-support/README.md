# Customer Support Agent

This sample demonstrates a customer support AI agent deployed on Google Cloud Platform using the Agent Development Kit (ADK) and Vertex AI.

## Features

- Order status retrieval from database
- Knowledge base search for product information
- Automatic escalation to human agents
- Session management with conversation history
- Integration with Google MCP servers

## Prerequisites

- Google Cloud account with billing enabled
- Project with Vertex AI API enabled
- gcloud CLI installed and configured
- Python 3.10 or later

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export PROJECT_ID=your-project-id
export LOCATION=us-central1
export MODEL_ID=gemini-2.0-flash
```

3. Authenticate with Google Cloud:
```bash
gcloud auth application-default login
```

## Running Locally

```bash
python main.py
```

## Deploying to Cloud Run

```bash
gcloud run deploy support-agent \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

## Deploying to Vertex AI Agent Engine

```bash
adk deploy agent_engine support_agent
```

## API Usage

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the status of order 12345?", "session_id": "test-session"}'
```

## Documentation

See the main deployment guide: [Google-GCP.md](../../Google-GCP.md)
