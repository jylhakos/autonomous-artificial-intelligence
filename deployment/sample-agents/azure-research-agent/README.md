# Research Assistant Agent

This sample demonstrates an academic research AI agent deployed on Microsoft Azure using AI Foundry and GPT-4 Turbo.

## Features

- Document search across Azure AI Search indexes
- Content analysis and summarization
- Research report generation
- Citation tracking and verification
- Multi-turn conversation support

## Prerequisites

- Azure subscription
- Azure AI Foundry project
- Azure OpenAI Service access
- Azure AI Search service
- Python 3.10 or later

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Azure credentials:
```bash
az login
```

3. Set environment variables:
```bash
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_KEY=your-api-key
export AZURE_OPENAI_DEPLOYMENT=gpt-4-turbo
export AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
export AZURE_SEARCH_KEY=your-search-key
export AZURE_SEARCH_INDEX=research-papers
```

## Running Locally

```bash
python main.py
```

## Deploying to Azure Container Apps

```bash
# Build and push container
az acr build -t research-agent:latest -r yourregistry .

# Deploy to Container Apps
az containerapp create \
    --name research-agent \
    --resource-group your-rg \
    --image yourregistry.azurecr.io/research-agent:latest \
    --environment your-env \
    --ingress external \
    --target-port 8000
```

## API Usage

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "quantum computing applications", "top_k": 5}'
```

## Documentation

See the main deployment guide: [Microsoft-Azure.md](../../Microsoft-Azure.md)
