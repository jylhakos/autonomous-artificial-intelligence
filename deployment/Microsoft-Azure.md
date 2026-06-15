# Microsoft Azure

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Virtual Environment Setup](#virtual-environment-setup)
4. [Azure Architecture Overview](#azure-architecture-overview)
5. [Deployment Options](#deployment-options)
6. [Use Case: Research Assistant Agent](#use-case-research-assistant-agent)
7. [Step-by-Step Implementation](#step-by-step-implementation)
8. [Advanced Features](#advanced-features)
9. [Retrieval-Augmented Generation (RAG) Evaluators and Chunking](#retrieval-augmented-generation-rag-evaluators-and-chunking)
10. [Security and Governance](#security-and-governance)
11. [Monitoring and Observability](#monitoring-and-observability)
12. [Cost Optimization](#cost-optimization)
13. [Troubleshooting](#troubleshooting)
14. [References](#references)

## Introduction

Microsoft Azure provides an AI agent platform that integrates seamlessly with the Microsoft ecosystem while offering enterprise-grade security, scalability, and observability. Azure AI Foundry, combined with the Agent Framework and Entra Agent ID authentication, enables organizations to build production-ready agentic systems with minimal infrastructure overhead.

### Key Components

**Azure AI Foundry**: A unified platform for developing, deploying, and managing AI agents with pre-built templates and integrated tooling.

**Agent Framework**: Microsoft's open-source framework for building multi-agent systems with support for orchestration, memory, and tool integration.

**Entra Agent ID**: Identity and access management specifically designed for AI agents, providing secure authentication and authorization.

**Azure OpenAI Service**: Access to GPT-4, GPT-4 Turbo, and other OpenAI models through Azure's compliance-certified infrastructure.

**Azure Container Apps**: Serverless container platform for deploying agents with automatic scaling and built-in networking.

### Why Choose Azure for AI Agents

1. **Microsoft Ecosystem Integration**: Native connectivity to Microsoft 365, Dynamics 365, Power Platform
2. **Enterprise Compliance**: Built-in support for GDPR, HIPAA, SOC 2, ISO 27001 across all regions
3. **Hybrid Cloud Flexibility**: Azure Arc enables running agents across cloud, on-premises, and edge environments
4. **Developer Productivity**: Visual Studio Code integration, GitHub Copilot support, Azure DevOps CI/CD
5. **Cost Management**: Reserved capacity options and detailed cost tracking with Azure Cost Management

## Prerequisites

Before beginning deployment, ensure you have:

### Required Accounts and Access

- An Azure subscription (Free tier or Pay-As-You-Go)
- Azure CLI 2.50+ installed and configured
- Appropriate role assignments (Contributor or Owner on resource group)
- GitHub account (for CI/CD deployments)

### Required Software

- Python 3.10 or later
- Azure CLI (`az`)
- Docker Desktop
- Visual Studio Code (recommended)
- Azure Functions Core Tools (for local testing)

### Service Registration

Register required resource providers:

```bash
# Register Azure OpenAI service
az provider register --namespace Microsoft.CognitiveServices

# Register Container Apps
az provider register --namespace Microsoft.App

# Register AI Foundry (preview)
az provider register --namespace Microsoft.AIFoundry

# Verify registration status
az provider show --namespace Microsoft.CognitiveServices --query "registrationState"
```

### Authentication

Log in to Azure:

```bash
# Interactive login
az login

# Set default subscription
az account set --subscription "Your Subscription Name"

# Verify current subscription
az account show
```

## Virtual Environment Setup

### Initial Setup

1. **Create a project directory**:

```bash
mkdir azure-research-agent
cd azure-research-agent
```

2. **Create and activate virtual environment**:

**On macOS/Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Upgrade pip**:

```bash
pip install --upgrade pip
```

### Install Dependencies

Create a `requirements.txt` file:

```txt
# Azure SDKs
azure-identity>=1.15.0
azure-ai-openai>=1.0.0
azure-cosmos>=4.5.0
azure-storage-blob>=12.19.0
azure-monitor-opentelemetry>=1.2.0
azure-core>=1.29.0

# Agent Framework
azure-agent-framework>=0.1.0
semantic-kernel>=1.0.0

# MCP integration
mcp>=0.1.0

# Web framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0

# Data processing
pydantic>=2.5.0
python-dotenv>=1.0.0
aiohttp>=3.9.0

# Search and retrieval
azure-search-documents>=11.4.0
langchain>=0.1.0
langchain-openai>=0.0.5

# Observability
opencensus-ext-azure>=1.1.9
applicationinsights>=0.11.10

# Development tools
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
ruff>=0.1.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```bash
# Azure Configuration
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=research-agent-rg
AZURE_LOCATION=eastus

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4-turbo
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your-search-key
AZURE_SEARCH_INDEX=research-documents

# Cosmos DB Configuration (for memory)
COSMOS_DB_ENDPOINT=https://your-cosmos-db.documents.azure.com:443/
COSMOS_DB_KEY=your-cosmos-key
COSMOS_DB_DATABASE=agent-memory
COSMOS_DB_CONTAINER=sessions

# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
PORT=8000
```

### VS Code Integration

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "azureFunctions.deploySubpath": ".",
    "azureFunctions.scmDoBuildDuringDeployment": true,
    "azureFunctions.pythonVenv": "venv",
    "azureFunctions.projectRuntime": "~4",
    "azureFunctions.projectLanguage": "Python"
}
```

## Azure Architecture Overview

### Component Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Client Applications                       │
│  (Microsoft Teams, Power Apps, Web Portal, Mobile App)       │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                   Azure Front Door / API Management          │
│        (Global load balancing, API gateway, WAF)             │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              Azure Container Apps / App Service              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │           Research Assistant Agent (FastAPI)           │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │      Agent Framework Orchestration               │  │  │
│  │  │  - Request routing and validation                │  │  │
│  │  │  - Tool selection and execution                  │  │  │
│  │  │  - Memory management (Cosmos DB)                 │  │  │
│  │  │  - Response synthesis                            │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
       │           │              │              │
       ▼           ▼              ▼              ▼
┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────┐
│  Azure   │ │   Azure    │ │ Cosmos   │ │ Application  │
│ OpenAI   │ │ AI Search  │ │    DB    │ │  Insights    │
│ Service  │ │            │ │          │ │              │
└──────────┘ └────────────┘ └──────────┘ └──────────────┘
     │
     ├─ GPT-4 Turbo
     ├─ GPT-4
     └─ text-embedding-ada-002
```

### Data Flow

1. **Request Ingestion**: User request via Teams bot, Power Apps, or direct API call
2. **Authentication**: Entra Agent ID validates identity and retrieves permissions
3. **Agent Orchestration**: Framework processes request and plans tool execution
4. **Tool Invocation**: Agent calls Azure AI Search, custom APIs, or Microsoft Graph
5. **LLM Reasoning**: Azure OpenAI (GPT-4) processes context and generates response
6. **Memory Update**: Session state persisted to Cosmos DB for conversation continuity
7. **Response Delivery**: Structured response returned with citations and confidence scores
8. **Telemetry**: Application Insights captures metrics, traces, and logs

### Integration with Microsoft Services

```
Research Agent
    ├─ Microsoft Graph API (email, calendar, OneDrive)
    ├─ SharePoint Online (document repositories)
    ├─ Microsoft Teams (bot interface)
    ├─ Power Automate (workflow triggers)
    ├─ Dynamics 365 (CRM data)
    └─ Azure DevOps (code repositories, work items)
```

## Deployment Options

### Option 1: Managed Deployment with Azure AI Foundry

**Best for**: Rapid prototyping and managed production deployments

**Advantages**:
- No infrastructure management required
- Built-in monitoring and logging
- Automatic scaling and load balancing
- Integrated security and compliance

**Deployment via Azure Developer CLI**:

```bash
# Install Azure Developer CLI
curl -fsSL https://aka.ms/install-azd.sh | bash

# Initialize from template
azd init --template azure-ai-agent-foundry

# Provision resources and deploy
azd up
```

### Option 2: Container Deployment with Azure Container Apps

**Best for**: Custom containerized agents with specific runtime requirements

**Advantages**:
- Full control over container image
- Support for sidecars and Dapr integration
- KEDA-based autoscaling
- Zero to N scaling

**Deployment**:

```bash
# Create resource group
az group create --name research-agent-rg --location eastus

# Create Container Apps environment
az containerapp env create \
    --name research-env \
    --resource-group research-agent-rg \
    --location eastus

# Build and deploy
az containerapp up \
    --name research-agent \
    --source . \
    --environment research-env \
    --resource-group research-agent-rg \
    --ingress external \
    --target-port 8000
```

### Option 3: Serverless with Azure Functions

**Best for**: Event-driven agents triggered by messages, timers, or webhooks

**Advantages**:
- Cost-effective for sporadic workloads
- Deep integration with Azure services
- Built-in bindings for queues, databases, etc.
- Consumption-based pricing

**Deployment**:

```bash
# Create Function App
az functionapp create \
    --name research-agent-func \
    --resource-group research-agent-rg \
    --consumption-plan-location eastus \
    --runtime python \
    --runtime-version 3.10 \
    --functions-version 4 \
    --storage-account researchagentstorage

# Deploy code
func azure functionapp publish research-agent-func
```

## Use Case: Research Assistant Agent

### Overview

An automated research assistant agent that helps users find relevant information across multiple sources, synthesize insights, and generate reports demonstrates Azure's integration capabilities with Microsoft 365 and AI Search.

### Capabilities

1. **Multi-Source Search**: Query Azure AI Search, SharePoint, academic databases
2. **Document Analysis**: Extract insights from PDFs, Word docs, PowerPoint presentations
3. **Citation Management**: Track sources and generate bibliographies
4. **Report Generation**: Create structured research reports in Word format
5. **Collaboration**: Share findings via Teams and email

### Business Requirements

- **Privacy**: User search queries and findings remain private within their tenant
- **Accuracy**: All claims must include citations to source documents
- **Performance**: Search results returned within 3 seconds
- **Accessibility**: Available via Teams, web portal, and API

### Sample Implementation

```python
"""Research Assistant Agent with Azure AI Search integration."""
from azure.identity import DefaultAzureCredential
from azure.ai.openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from typing import List, Dict, Any
import os

class ResearchAssistantAgent:
    """AI agent for automated research and information synthesis."""
    
    def __init__(self):
        """Initialize agent with Azure services."""
        self.credential = DefaultAzureCredential()
        
        # Initialize Azure OpenAI client
        self.openai_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        
        # Initialize Azure AI Search client
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX"),
            credential=self.credential
        )
        
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        # Generate query embedding
        embedding_response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_vector = embedding_response.data[0].embedding
        
        # Perform hybrid search (keyword + vector)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="content_vector"
        )
        
        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=["title", "content", "source", "url", "published_date"],
            top=top_k
        )
        
        documents = []
        for result in results:
            documents.append({
                "title": result["title"],
                "content": result["content"],
                "source": result["source"],
                "url": result.get("url"),
                "published_date": result.get("published_date"),
                "relevance_score": result["@search.score"]
            })
        
        return documents
    
    def analyze_documents(self, documents: List[Dict], research_question: str) -> str:
        """Analyze documents and synthesize insights.
        
        Args:
            documents: List of relevant documents
            research_question: Original research question
            
        Returns:
            Synthesized analysis with citations
        """
        # Prepare context from documents
        context = "\n\n".join([
            f"[{i+1}] {doc['title']} ({doc['source']})\n{doc['content'][:500]}..."
            for i, doc in enumerate(documents)
        ])
        
        # Create analysis prompt
        messages = [
            {
                "role": "system",
                "content": """You are an expert research analyst. Your task is to:
1. Analyze the provided documents
2. Answer the research question with evidence from the sources
3. Include citations using [1], [2], etc. notation
4. Highlight any conflicting information or research gaps
5. Provide a balanced, objective analysis"""
            },
            {
                "role": "user",
                "content": f"""Research Question: {research_question}

Available Sources:
{context}

Please provide an analysis with citations."""
            }
        ]
        
        # Generate analysis
        response = self.openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def generate_report(self, research_question: str, analysis: str, documents: List[Dict]) -> Dict[str, Any]:
        """Generate a structured research report.
        
        Args:
            research_question: Original research question
            analysis: Synthesized analysis
            documents: Source documents
            
        Returns:
            Structured report with bibliography
        """
        # Format bibliography
        bibliography = "\n".join([
            f"[{i+1}] {doc['title']}. {doc['source']}. {doc.get('published_date', 'n.d.')}. {doc.get('url', '')}"
            for i, doc in enumerate(documents)
        ])
        
        report = {
            "title": f"Research Report: {research_question}",
            "research_question": research_question,
            "executive_summary": analysis.split('\n')[0],
            "detailed_analysis": analysis,
            "sources_consulted": len(documents),
            "bibliography": bibliography,
            "generated_date": datetime.now().isoformat()
        }
        
        return report
    
    async def run(self, research_question: str) -> Dict[str, Any]:
        """Execute complete research workflow.
        
        Args:
            research_question: User's research question
            
        Returns:
            Complete research report
        """
        # Step 1: Search for relevant documents
        documents = self.search_documents(research_question, top_k=10)
        
        # Step 2: Analyze and synthesize findings
        analysis = self.analyze_documents(documents, research_question)
        
        # Step 3: Generate structured report
        report = self.generate_report(research_question, analysis, documents)
        
        return report
```

## Step-by-Step Implementation

### Step 1: Create Azure Resources

Create a script `infrastructure/setup.sh`:

```bash
#!/bin/bash

# Variables
RESOURCE_GROUP="research-agent-rg"
LOCATION="eastus"
OPENAI_ACCOUNT="research-openai-$(date +%s)"
SEARCH_SERVICE="research-search-$(date +%s)"
COSMOS_ACCOUNT="research-cosmos-$(date +%s)"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure OpenAI resource
az cognitiveservices account create \
    --name $OPENAI_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --kind OpenAI \
    --sku S0 \
    --yes

# Deploy GPT-4 model
az cognitiveservices account deployment create \
    --name $OPENAI_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --deployment-name gpt-4-turbo \
    --model-name gpt-4 \
    --model-version "turbo-2024-04-09" \
    --model-format OpenAI \
    --sku-capacity 10 \
    --sku-name "Standard"

# Create Azure AI Search service
az search service create \
    --name $SEARCH_SERVICE \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku standard

# Create Cosmos DB account
az cosmosdb create \
    --name $COSMOS_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --locations regionName=$LOCATION \
    --default-consistency-level Session

# Create database and container
az cosmosdb sql database create \
    --account-name $COSMOS_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --name agent-memory

az cosmosdb sql container create \
    --account-name $COSMOS_ACCOUNT \
    --database-name agent-memory \
    --resource-group $RESOURCE_GROUP \
    --name sessions \
    --partition-key-path "/userId" \
    --throughput 400

echo "Resources created successfully!"
echo "OpenAI Account: $OPENAI_ACCOUNT"
echo "Search Service: $SEARCH_SERVICE"
echo "Cosmos DB: $COSMOS_ACCOUNT"
```

Run the setup script:

```bash
chmod +x infrastructure/setup.sh
./infrastructure/setup.sh
```

### Step 2: Configure Azure AI Search Index

Create `infrastructure/search-index.json`:

```json
{
  "name": "research-documents",
  "fields": [
    {
      "name": "id",
      "type": "Edm.String",
      "key": true,
      "searchable": false
    },
    {
      "name": "title",
      "type": "Edm.String",
      "searchable": true,
      "filterable": true
    },
    {
      "name": "content",
      "type": "Edm.String",
      "searchable": true
    },
    {
      "name": "content_vector",
      "type": "Collection(Edm.Single)",
      "searchable": true,
      "dimensions": 1536,
      "vectorSearchProfile": "default-vector-config"
    },
    {
      "name": "source",
      "type": "Edm.String",
      "filterable": true,
      "facetable": true
    },
    {
      "name": "url",
      "type": "Edm.String",
      "searchable": false
    },
    {
      "name": "published_date",
      "type": "Edm.DateTimeOffset",
      "filterable": true,
      "sortable": true
    }
  ],
  "vectorSearch": {
    "profiles": [
      {
        "name": "default-vector-config",
        "algorithm": "hnsw-config"
      }
    ],
    "algorithms": [
      {
        "name": "hnsw-config",
        "kind": "hnsw",
        "hnswParameters": {
          "metric": "cosine",
          "m": 4,
          "efConstruction": 400,
          "efSearch": 500
        }
      }
    ]
  }
}
```

Create the index:

```bash
# Get search admin key
SEARCH_KEY=$(az search admin-key show \
    --service-name $SEARCH_SERVICE \
    --resource-group $RESOURCE_GROUP \
    --query primaryKey -o tsv)

# Create index
curl -X PUT \
    "https://$SEARCH_SERVICE.search.windows.net/indexes/research-documents?api-version=2023-11-01" \
    -H "Content-Type: application/json" \
    -H "api-key: $SEARCH_KEY" \
    -d @infrastructure/search-index.json
```

### Step 3: Implement Agent Framework

Create `src/agent/research_agent.py`:

```python
"""Research agent implementation with Azure services."""
from azure.identity import DefaultAzureCredential
from azure.cosmos.aio import CosmosClient
from azure.search.documents.aio import SearchClient
from azure.ai.openai import AzureOpenAI
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manage conversation memory with Cosmos DB."""
    
    def __init__(self, cosmos_endpoint: str, database: str, container: str):
        self.client = CosmosClient(cosmos_endpoint, DefaultAzureCredential())
        self.database = database
        self.container = container
    
    async def save_conversation(self, user_id: str, session_id: str, messages: List[Dict]):
        """Save conversation history."""
        container = self.client.get_database_client(self.database).get_container_client(self.container)
        
        document = {
            "id": session_id,
            "userId": user_id,
            "messages": messages,
            "lastUpdated": datetime.utcnow().isoformat()
        }
        
        await container.upsert_item(document)
    
    async def get_conversation(self, user_id: str, session_id: str) -> Optional[List[Dict]]:
        """Retrieve conversation history."""
        container = self.client.get_database_client(self.database).get_container_client(self.container)
        
        try:
            response = await container.read_item(item=session_id, partition_key=user_id)
            return response.get("messages", [])
        except Exception as e:
            logger.warning(f"No conversation found: {e}")
            return []

class ResearchAgent:
    """Complete research agent with memory and tool orchestration."""
    
    def __init__(self, config: dict):
        self.config = config
        self.openai = AzureOpenAI(
            azure_endpoint=config["openai_endpoint"],
            api_key=config["openai_key"],
            api_version=config["openai_api_version"]
        )
        self.search = SearchClient(
            endpoint=config["search_endpoint"],
            index_name=config["search_index"],
            credential=DefaultAzureCredential()
        )
        self.memory = MemoryManager(
            cosmos_endpoint=config["cosmos_endpoint"],
            database=config["cosmos_database"],
            container=config["cosmos_container"]
        )
    
    async def process_request(
        self,
        user_id: str,
        session_id: str,
        message: str
    ) -> Dict[str, Any]:
        """Process user research request end-to-end."""
        
        # Retrieve conversation history
        history = await self.memory.get_conversation(user_id, session_id)
        
        # Search documents
        documents = await self._search_documents(message)
        
        # Generate response with citations
        response, references = await self._generate_response(message, documents, history)
        
        # Update conversation history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        await self.memory.save_conversation(user_id, session_id, history)
        
        return {
            "response": response,
            "references": references,
            "session_id": session_id
        }
    
    async def _search_documents(self, query: str) -> List[Dict]:
        """Search implementation."""
        # Search logic here
        pass
    
    async def _generate_response(
        self,
        query: str,
        documents: List[Dict],
        history: List[Dict]
    ) -> tuple[str, List[Dict]]:
        """Generate response with citations."""
        # Response generation logic
        pass
```

### Step 4: Create FastAPI Application

Create `src/main.py`:

```python
"""FastAPI application for research agent."""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from azure.identity import DefaultAzureCredential
import os
import logging
from src.agent.research_agent import ResearchAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="Research Assistant Agent API",
    description="AI-powered research and information synthesis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize agent
config = {
    "openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "openai_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    "search_endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
    "search_index": os.getenv("AZURE_SEARCH_INDEX"),
    "cosmos_endpoint": os.getenv("COSMOS_DB_ENDPOINT"),
    "cosmos_database": os.getenv("COSMOS_DB_DATABASE"),
    "cosmos_container": os.getenv("COSMOS_DB_CONTAINER")
}

agent = ResearchAgent(config)

class ResearchRequest(BaseModel):
    """Research request model."""
    query: str
    user_id: str
    session_id: str

class ResearchResponse(BaseModel):
    """Research response model."""
    response: str
    references: list
    session_id: str

@app.post("/research", response_model=ResearchResponse)
async def conduct_research(request: ResearchRequest):
    """Process research request.
    
    Args:
        request: Research request with query and session info
        
    Returns:
        Research findings with citations
    """
    try:
        logger.info(f"Processing research request: {request.query}")
        
        result = await agent.process_request(
            user_id=request.user_id,
            session_id=request.session_id,
            message=request.query
        )
        
        return ResearchResponse(**result)
        
    except Exception as e:
        logger.error(f"Research error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "research-agent",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 5: Create Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 6: Deploy to Azure Container Apps

Create `infrastructure/deploy.sh`:

```bash
#!/bin/bash

RESOURCE_GROUP="research-agent-rg"
APP_NAME="research-agent"
ENVIRONMENT="research-env"

# Build and push container image
az acr build \
    --registry researchagentacr \
    --resource-group $RESOURCE_GROUP \
    --image $APP_NAME:latest \
    .

# Deploy to Container Apps
az containerapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $ENVIRONMENT \
    --image researchagentacr.azurecr.io/$APP_NAME:latest \
    --target-port 8000 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 10 \
    --cpu 1.0 \
    --memory 2Gi \
    --env-vars \
        AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT \
        AZURE_SEARCH_ENDPOINT=$AZURE_SEARCH_ENDPOINT \
        COSMOS_DB_ENDPOINT=$COSMOS_DB_ENDPOINT

echo "Deployment complete!"
```

Run deployment:

```bash
chmod +x infrastructure/deploy.sh
./infrastructure/deploy.sh
```

## Advanced Features

### Microsoft Graph Integration

Access Microsoft 365 data within agents:

```python
from msgraph import GraphServiceClient
from azure.identity import DefaultAzureCredential

class GraphToolset:
    """Tools for accessing Microsoft 365 data."""
    
    def __init__(self):
        credential = DefaultAzureCredential()
        self.client = GraphServiceClient(credentials=credential)
    
    async def search_emails(self, query: str) -> List[Dict]:
        """Search user's emails."""
        result = await self.client.me.messages.get(
            filter=f"contains(subject, '{query}')"
        )
        return [{"subject": msg.subject, "body": msg.body.content} 
                for msg in result.value]
    
    async def get_sharepoint_documents(self, site_id: str, query: str) -> List[Dict]:
        """Search SharePoint documents."""
        result = await self.client.sites.by_site_id(site_id).drive.search(query).get()
        return [{"name": item.name, "webUrl": item.web_url} 
                for item in result.value]
```

### Teams Bot Integration

Deploy agent as a Microsoft Teams bot:

```python
from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount

class ResearchBotActivityHandler(ActivityHandler):
    """Teams bot handler for research agent."""
    
    def __init__(self, research_agent: ResearchAgent):
        self.agent = research_agent
    
    async def on_message_activity(self, turn_context: TurnContext):
        """Handle incoming messages from Teams."""
        user_message = turn_context.activity.text
        user_id = turn_context.activity.from_property.id
        conversation_id = turn_context.activity.conversation.id
        
        # Process through research agent
        result = await self.agent.process_request(
            user_id=user_id,
            session_id=conversation_id,
            message=user_message
        )
        
        # Send response back to Teams
        await turn_context.send_activity(result["response"])
```

### Semantic Kernel Integration

Use Semantic Kernel for advanced orchestration:

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

# Initialize kernel
kernel = sk.Kernel()

# Add Azure OpenAI service
kernel.add_chat_service(
    "chat",
    AzureChatCompletion(
        deployment_name="gpt-4-turbo",
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
)

# Define semantic function
research_function = kernel.create_semantic_function(
    """Given this research question: {{$input}}
    
    Search for relevant information and provide an answer with citations.
    Focus on recent and authoritative sources.""",
    max_tokens=2000,
    temperature=0.7
)

# Execute
result = await research_function.invoke_async("What are the latest developments in quantum computing?")
```

## Retrieval-Augmented Generation (RAG) Evaluators and Chunking

### Overview

Retrieval-Augmented Generation (RAG) is a powerful technique that combines the strengths of large language models with external knowledge retrieval systems. Azure AI Foundry provides tools and evaluators to assess and optimize RAG systems, ensuring they deliver accurate, relevant, and grounded responses.

A RAG system retrieves relevant information from a knowledge base to provide context for generating responses. The quality of a RAG system depends on two critical phases:

1. **Retrieval Quality**: How well the system finds relevant information
2. **Generation Quality**: How well the system uses that information to create accurate responses

### What is RAG Chunking?

**Chunking** is the process of breaking down large documents into smaller, manageable pieces called chunks. This is a crucial preprocessing step when preparing data for use with Large Language Models (LLMs) and vector databases.

In simple terms, chunking divides text into semantic units that are:
- **Small enough** to fit within model context windows
- **Large enough** to contain meaningful, contextually complete information
- **Optimally sized** for accurate retrieval and generation

Think of chunks as the atomic units of information that your RAG system works with. Each chunk should make sense on its own without requiring extensive surrounding context.

### Why is Chunking Important for RAG?

Chunking serves three critical functions in RAG systems, particularly for platforms like Weaviate and Pinecone:

#### 1. Fits Within LLM Context Windows

Language models have a maximum token limit for processing input. You cannot pass a 50-page document into a prompt all at once. 

**Why this matters:**
- LLMs like GPT-4 have context windows ranging from 8K to 128K tokens
- Embedding models have even smaller limits (often 512-8192 tokens)
- Exceeding these limits causes truncation, losing critical information

**Solution:** Chunking breaks documents down into manageable sizes that fit securely within the model's memory, ensuring all information can be processed effectively.

#### 2. Maximizes Search Precision

When an embedding model converts a block of text into a vector, the text's meaning is condensed into a high-dimensional representation.

**The Challenge:**
- **Chunks too large** (like an entire chapter): The specific meaning gets diminished. Multiple ideas mixed together create a noisy, "averaged" embedding that doesn't clearly represent any single topic
- **Chunks too small** (like a single sentence): You lose necessary context and relationships between ideas

**Solution:** Optimal chunking balances the two, guaranteeing the search engine finds specific, highly relevant answers. Well-formed chunks enable:
- More precise vector similarity matching
- Better semantic search results
- Improved retrieval of contextually appropriate information

#### 3. Reduces AI Hallucinations

When an AI receives too much irrelevant or oversized information, it struggles to locate the right facts, which leads to hallucinations (fabricated information).

**How chunking helps:**
- By feeding the LLM tightly scoped, relevant chunks, the model is grounded with the specific facts it needs
- Smaller, focused chunks reduce the "lost in the middle" problem where models miss important information buried in long contexts
- Precise chunks provide clear signal-to-noise ratio, minimizing confabulation

**Result:** The model generates accurate responses based on factual, well-contextualized information rather than inventing details.

### RAG Evaluators in Azure AI Foundry

Azure provides two categories of evaluators for RAG systems:

#### System Evaluation

System evaluation examines the quality of the final response in your RAG workflow:

**Groundedness Evaluator**
- **Purpose**: Measures if the response is grounded in the provided context without fabrication
- **Score**: 1-5 scale (default threshold: 3)
- **Use case**: Ensures precision - responses don't contain content outside grounding context

```python
testing_criteria = [
    {
        "type": "azure_ai_evaluator",
        "name": "groundedness",
        "evaluator_name": "builtin.groundedness",
        "initialization_parameters": {"deployment_name": model_deployment},
        "data_mapping": {
            "context": "{{item.context}}",
            "response": "{{item.response}}",
            "query": "{{item.query}}"  # Optional but improves accuracy
        },
    }
]
```

**Groundedness Pro Evaluator (Preview)**
- **Purpose**: Strict consistency check using Azure AI Content Safety service
- **Score**: Boolean (True/False)
- **Use case**: Enterprise applications requiring highest accuracy standards

**Relevance Evaluator**
- **Purpose**: Measures how accurately the response addresses the user's query
- **Score**: 1-5 scale
- **Use case**: Ensures responses are on-topic and directly answer the question

```python
{
    "type": "azure_ai_evaluator",
    "name": "relevance",
    "evaluator_name": "builtin.relevance",
    "initialization_parameters": {"deployment_name": model_deployment},
    "data_mapping": {
        "query": "{{item.query}}", 
        "response": "{{item.response}}"
    },
}
```

**Response Completeness Evaluator (Preview)**
- **Purpose**: Ensures responses don't miss critical information from ground truth
- **Score**: 1-5 scale
- **Use case**: Focuses on recall - responses cover all expected information

#### Process Evaluation

Process evaluation assesses the quality of document retrieval:

**Retrieval Evaluator**
- **Purpose**: Measures how relevant retrieved context chunks are to the query
- **Score**: Pass/Fail based on threshold
- **Use case**: Evaluates textual quality without ground truth labels

```python
{
    "type": "azure_ai_evaluator",
    "name": "retrieval",
    "evaluator_name": "builtin.retrieval",
    "initialization_parameters": {"deployment_name": model_deployment},
    "data_mapping": {
        "query": "{{item.query}}", 
        "context": "{{item.context}}"
    },
}
```

**Document Retrieval Evaluator**
- **Purpose**: The search quality metrics with ground truth
- **Metrics**: Fidelity, NDCG, XDCG, Max Relevance, Holes
- **Use case**: Precise measurement for debugging and parameter optimization

### Chunking Strategies

Different chunking strategies suit different document types and use cases:

#### Fixed-Size Chunking

The simplest approach - split text into predetermined chunk sizes with optional overlap.

**Characteristics:**
- **Complexity**: Low
- **Cost**: Low
- **Best for**: Quick prototyping, simple documents, baseline testing

```python
def fixed_size_chunking(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into fixed-size chunks with overlap."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

# Example usage
document = "Your large document text here..."
chunks = fixed_size_chunking(document, chunk_size=512, overlap=100)
```

**Pinecone Recommendation:**
- Use sliding windows with overlap (10-20% of chunk size)
- Choose chunk size aligned with embedding model's context window
- Use BERT tokens instead of character counts for better semantic preservation

#### Recursive Chunking

Splits text using a prioritized list of separators, recursively applying them until chunks fit the desired size.

**Characteristics:**
- **Complexity**: Medium
- **Cost**: Low
- **Best for**: Structured documents with natural separators

```python
def recursive_chunking(text: str, max_chunk_size: int = 1000) -> List[str]:
    """Recursively chunk text using prioritized separators."""
    if len(text) <= max_chunk_size:
        return [text.strip()] if text.strip() else []
    
    separators = ["\n\n", "\n", ". ", " "]
    
    for separator in separators:
        if separator in text:
            parts = text.split(separator)
            chunks = []
            current_chunk = ""
            
            for part in parts:
                test_chunk = current_chunk + separator + part if current_chunk else part
                
                if len(test_chunk) <= max_chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = part
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Recursively process large chunks
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > max_chunk_size:
                    final_chunks.extend(recursive_chunking(chunk, max_chunk_size))
                else:
                    final_chunks.append(chunk)
            
            return [chunk for chunk in final_chunks if chunk]
    
    return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
```

#### Semantic Chunking

Breaks text based on semantic similarity between sections, creating chunks grouped by meaning.

**Characteristics:**
- **Complexity**: High
- **Cost**: High (requires embeddings)
- **Best for**: Dense unstructured text, academic papers, legal documents

**Weaviate's Approach:**
1. Break document into sentences
2. Generate embeddings for each sentence with surrounding context
3. Compare semantic similarity between adjacent groups
4. Create chunk boundaries where semantic shifts occur

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunking(text: str, similarity_threshold: float = 0.5) -> List[str]:
    """Chunk text based on semantic similarity."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Split into sentences
    sentences = text.split('. ')
    
    # Generate embeddings
    embeddings = model.encode(sentences)
    
    # Calculate similarities
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = np.dot(embeddings[i], embeddings[i-1])
        
        if similarity > similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentences[i]]
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks
```

#### Document Structure-Based Chunking

Uses the intrinsic structure of documents (headers, sections, etc.) to create chunks.

**Characteristics:**
- **Complexity**: Medium
- **Cost**: Low-Medium
- **Best for**: Markdown, HTML, structured documents

```python
import re

def markdown_chunking(text: str) -> List[str]:
    """Chunk Markdown documents by headers."""
    header_pattern = r'^#{1,6}\s+.+$'
    lines = text.split('\n')
    
    chunks = []
    current_chunk = []
    
    for line in lines:
        if re.match(header_pattern, line, re.MULTILINE):
            if current_chunk:
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
            current_chunk = [line]
        else:
            current_chunk.append(line)
    
    if current_chunk:
        chunk_text = '\n'.join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)
    
    return chunks
```

#### Late Chunking (Advanced)

Creates token-level embeddings for entire documents first, then averages relevant tokens for each chunk.

**Weaviate's Innovation:**
- Preserves full document context
- Chunks retain relationships to other parts of document
- Ideal for technical documents with cross-references

**Use case**: Research papers, legal texts where sections reference each other

#### Contextual Chunking (Anthropic Method)

Uses LLM to generate contextual descriptions for chunks, preserving high-level document context.

```python
async def contextual_chunking(document: str, chunks: List[str], llm_client) -> List[str]:
    """Add context to chunks using LLM."""
    contextualized_chunks = []
    
    # Generate document summary
    summary = await llm_client.generate(
        f"Summarize this document in 2-3 sentences:\n\n{document[:2000]}"
    )
    
    for chunk in chunks:
        # Generate contextual description
        context = await llm_client.generate(
            f"Document context: {summary}\n\n"
            f"Provide a brief context (1 sentence) for this chunk:\n\n{chunk}"
        )
        
        # Prepend context to chunk
        contextualized_chunk = f"{context}\n\n{chunk}"
        contextualized_chunks.append(contextualized_chunk)
    
    return contextualized_chunks
```

### Azure-Specific Chunking Implementation

#### Using Azure Document Intelligence

```python
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

def azure_document_chunking(document_url: str) -> List[Dict]:
    """Extract and chunk documents using Azure Document Intelligence."""
    endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
    key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
    
    client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))
    
    poller = client.begin_analyze_document_from_url(
        "prebuilt-layout", document_url
    )
    result = poller.result()
    
    chunks = []
    for page in result.pages:
        for line in page.lines:
            chunks.append({
                "content": line.content,
                "page": page.page_number,
                "bounding_box": line.polygon
            })
    
    return chunks
```

#### Integrating with Azure AI Search

```python
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)

def create_chunked_index():
    """Create Azure AI Search index optimized for chunked documents."""
    index_client = SearchIndexClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
    )
    
    fields = [
        SearchField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
        SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                   searchable=True, vector_search_dimensions=1536,
                   vector_search_profile_name="myHnswProfile"),
        SearchField(name="document_id", type=SearchFieldDataType.String, filterable=True),
        SearchField(name="chunk_number", type=SearchFieldDataType.Int32, filterable=True),
        SearchField(name="document_title", type=SearchFieldDataType.String, searchable=True)
    ]
    
    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")],
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")]
    )
    
    index = SearchIndex(name="chunked-documents", fields=fields, vector_search=vector_search)
    index_client.create_or_update_index(index)
```

### Best Practices for Chunking

#### 1. Start with Fixed-Size Chunking

Begin with a baseline approach using 512-1024 tokens with 10-20% overlap. Test and iterate from there.

#### 2. Match Chunk Size to Use Case

- **Semantic search**: 256-512 tokens
- **Question answering**: 512-1024 tokens
- **Document summarization**: 1024-2048 tokens

#### 3. Consider Your Embedding Model

Different models have different optimal chunk sizes:
- `text-embedding-ada-002`: Up to 8191 tokens
- `text-embedding-3-small`: Up to 8191 tokens
- `text-embedding-3-large`: Up to 8191 tokens

#### 4. Preserve Metadata

Always include metadata with chunks:

```python
chunk_metadata = {
    "document_id": "doc_123",
    "chunk_number": 1,
    "source_url": "https://example.com/document",
    "created_at": "2026-06-15",
    "total_chunks": 10
}
```

#### 5. Use Structured IDs (Pinecone Pattern)

```python
# Format: document_id#chunk_number
chunk_id = f"document_{doc_id}#chunk_{chunk_num}"
```

This enables:
- Efficient chunk retrieval by document
- Easy updates and deletions
- Clear debugging and monitoring

#### 6. Monitor and Iterate

Track key metrics:
- **Retrieval precision**: Are correct chunks being retrieved?
- **Retrieval recall**: Are all relevant chunks being found?
- **Response quality**: Are generated answers accurate and complete?
- **Latency**: Is query time acceptable?

```python
from azure.monitor.opentelemetry import configure_azure_monitor

configure_azure_monitor()

# Track chunking metrics
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("chunk_retrieval") as span:
    span.set_attribute("chunk_count", len(chunks))
    span.set_attribute("query", user_query)
    # ... retrieval logic
```

### Complete RAG Pipeline Example

```python
from typing import List, Dict
from openai import AzureOpenAI
from azure.search.documents import SearchClient
import os

class AzureRAGPipeline:
    """Complete RAG pipeline with optimized chunking."""
    
    def __init__(self):
        self.openai_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview"
        )
        
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name="chunked-documents",
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
        )
    
    def chunk_document(self, document: str, chunk_size: int = 512) -> List[str]:
        """Chunk document using recursive strategy."""
        return recursive_chunking(document, chunk_size)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Azure OpenAI."""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    async def index_document(self, document: str, document_id: str, title: str):
        """Chunk and index a document."""
        chunks = self.chunk_document(document)
        
        documents = []
        for i, chunk in enumerate(chunks):
            embedding = self.generate_embedding(chunk)
            
            documents.append({
                "id": f"{document_id}#chunk{i}",
                "content": chunk,
                "embedding": embedding,
                "document_id": document_id,
                "chunk_number": i,
                "document_title": title
            })
        
        self.search_client.upload_documents(documents)
    
    async def retrieve_and_generate(self, query: str, top_k: int = 3) -> Dict:
        """Retrieve relevant chunks and generate response."""
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search for relevant chunks
        results = self.search_client.search(
            search_text=None,
            vector_queries=[{
                "vector": query_embedding,
                "k": top_k,
                "fields": "embedding"
            }],
            select=["content", "document_title", "chunk_number"]
        )
        
        # Collect context from retrieved chunks
        context_parts = []
        sources = []
        
        for result in results:
            context_parts.append(result["content"])
            sources.append({
                "title": result["document_title"],
                "chunk": result["chunk_number"]
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate response with context
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.7
        )
        
        return {
            "response": response.choices[0].message.content,
            "sources": sources,
            "context": context
        }

# Usage example
rag_pipeline = AzureRAGPipeline()

# Index a document
await rag_pipeline.index_document(
    document="Your long document text here...",
    document_id="doc_001",
    title="Introduction to RAG"
)

# Query the system
result = await rag_pipeline.retrieve_and_generate(
    query="What is RAG and why is it important?"
)

print(result["response"])
```

### Evaluating Your RAG System

Combine chunking strategies with Azure evaluators:

```python
async def evaluate_rag_system(test_queries: List[Dict]):
    """Evaluate RAG system with Azure evaluators."""
    testing_criteria = [
        {
            "type": "azure_ai_evaluator",
            "name": "groundedness",
            "evaluator_name": "builtin.groundedness",
            "initialization_parameters": {"deployment_name": "gpt-4-turbo"},
            "data_mapping": {
                "context": "{{item.context}}",
                "response": "{{item.response}}",
                "query": "{{item.query}}"
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "relevance",
            "evaluator_name": "builtin.relevance",
            "initialization_parameters": {"deployment_name": "gpt-4-turbo"},
            "data_mapping": {
                "query": "{{item.query}}", 
                "response": "{{item.response}}"
            },
        },
        {
            "type": "azure_ai_evaluator",
            "name": "retrieval",
            "evaluator_name": "builtin.retrieval",
            "initialization_parameters": {"deployment_name": "gpt-4-turbo"},
            "data_mapping": {
                "query": "{{item.query}}", 
                "context": "{{item.context}}"
            },
        }
    ]
    
    results = []
    for query_data in test_queries:
        rag_result = await rag_pipeline.retrieve_and_generate(query_data["query"])
        
        evaluation = {
            "query": query_data["query"],
            "response": rag_result["response"],
            "context": rag_result["context"],
            "groundedness_score": None,
            "relevance_score": None,
            "retrieval_score": None
        }
        
        # Run evaluators (simplified)
        # In production, use Azure AI Foundry evaluation service
        
        results.append(evaluation)
    
    return results
```

### Key Takeaways

1. **Chunking is Essential**: Proper chunking ensures LLMs receive the right amount of context for accurate responses

2. **Three Critical Functions**:
   - Fits within context windows
   - Maximizes search precision
   - Reduces hallucinations

3. **Multiple Strategies Available**: Choose based on document type, use case, and quality requirements

4. **Platform-Specific Optimizations**:
   - **Azure**: Document Intelligence, AI Search integration
   - **Pinecone**: Structured IDs, metadata filtering
   - **Weaviate**: Late chunking, semantic chunking

5. **Evaluation is Key**: Use Azure's built-in evaluators to measure and improve RAG performance

6. **Iterate and Optimize**: Start simple (fixed-size), measure results, then enhance based on specific needs

### Additional Resources

- [Azure RAG Evaluators Documentation](https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/rag-evaluators)
- [Azure RAG Chunking Phase Guide](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase)
- [Weaviate Chunking Strategies](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Pinecone Chunking Best Practices](https://www.pinecone.io/learn/chunking-strategies/)
- [Pinecone Data Modeling Guide](https://docs.pinecone.io/guides/index-data/data-modeling)

## Security and Governance

### Entra ID Authentication

Secure API with Azure AD:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from azure.identity import DefaultAzureCredential
import jwt

security = HTTPBearer()

async def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate Azure AD token."""
    token = credentials.credentials
    
    try:
        # Decode and validate JWT
        decoded = jwt.decode(
            token,
            options={"verify_signature": False},  # Verify with Azure AD public keys in production
            audience=os.getenv("AZURE_CLIENT_ID")
        )
        return decoded
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

@app.post("/research", dependencies=[Depends(validate_token)])
async def conduct_research(request: ResearchRequest):
    # Protected endpoint
    pass
```

### Data Encryption

Encrypt sensitive data in Cosmos DB:

```python
from cryptography.fernet import Fernet
import os

class EncryptionService:
    """Encrypt/decrypt sensitive data."""
    
    def __init__(self):
        key = os.getenv("ENCRYPTION_KEY").encode()
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
```

### Role-Based Access Control

Implement RBAC for agent features:

```python
from enum import Enum

class UserRole(Enum):
    READER = "reader"
    RESEARCHER = "researcher"
    ADMIN = "admin"

def require_role(required_role: UserRole):
    """Decorator to enforce role-based access."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user role from token
            user_role = kwargs.get("user_role")
            if user_role != required_role:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@app.post("/admin/reindex")
@require_role(UserRole.ADMIN)
async def reindex_documents():
    """Admin-only endpoint to trigger reindexing."""
    pass
```

## Monitoring and Observability

### Application Insights Integration

```python
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.trace.tracer import Tracer

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(
    connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
))

# Configure tracing
tracer = Tracer(
    exporter=AzureExporter(
        connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    ),
    sampler=ProbabilitySampler(1.0)
)

# Use in endpoints
@app.post("/research")
async def conduct_research(request: ResearchRequest):
    with tracer.span(name="research_request"):
        logger.info(f"Research query: {request.query}")
        result = await agent.process_request(...)
        logger.info(f"Research completed in {elapsed_time}ms")
        return result
```

### Custom Metrics

```python
from opencensus.stats import aggregation as aggregation_module
from opencensus.stats import measure as measure_module
from opencensus.stats import stats as stats_module
from opencensus.stats import view as view_module

# Define measures
research_duration_measure = measure_module.MeasureFloat(
    "research/duration",
    "Duration of research requests",
    "ms"
)

token_usage_measure = measure_module.MeasureInt(
    "research/tokens_used",
    "Number of tokens consumed",
    "tokens"
)

# Record metrics
stats = stats_module.stats
mmap = stats.stats_recorder

with mmap.measure_float_put(research_duration_measure, elapsed_ms):
    pass
```

## Cost Optimization

### Pricing Calculator

| Component | Configuration | Monthly Cost (Estimate) |
|-----------|--------------|-------------------------|
| Azure OpenAI (GPT-4) | 1M tokens/month | $30-60 |
| Azure AI Search (Standard) | 1 replica, 1 partition | $250 |
| Cosmos DB | 400 RU/s, 10GB | $24 |
| Container Apps | 1 vCPU, 2 GB RAM, 1M requests | $50-100 |
| Application Insights | 5GB logs/month | $10 |
| **Total** | | **$364-444/month** |

### Cost Reduction Strategies

```python
# Implement caching to reduce OpenAI calls
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str) -> List[float]:
    """Cache embeddings for frequently queried text."""
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Batching for efficiency
async def process_batch(queries: List[str]) -> List[Dict]:
    """Process multiple queries in single API call."""
    tasks = [search_documents(q) for q in queries]
    return await asyncio.gather(*tasks)
```

## Troubleshooting

### Common Issues

**Issue**: `401 Unauthorized` when calling Azure OpenAI

**Solution**:
```bash
# Verify authentication
az account show

# Check role assignments
az role assignment list \
    --assignee $(az ad signed-in-user show --query id -o tsv) \
    --scope /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP

# Grant Cognitive Services User role
az role assignment create \
    --assignee $(az ad signed-in-user show --query id -o tsv) \
    --role "Cognitive Services User" \
    --scope /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP
```

**Issue**: Slow search performance

**Solution**:
```python
# Optimize search with filters
results = search_client.search(
    search_text=query,
    filter=f"published_date ge {recent_date}",  # Filter to recent docs
    select=["title", "content"],  # Only retrieve needed fields
    top=5  # Limit results
)
```

**Issue**: High token consumption

**Solution**:
```python
# Truncate prompts intelligently
def truncate_context(context: str, max_tokens: int = 3000) -> str:
    """Truncate context to fit token limit."""
    tokens = context.split()
    if len(tokens) > max_tokens:
        return ' '.join(tokens[:max_tokens]) + "... [truncated]"
    return context
```

### Debug Mode

Enable verbose logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_debug.log'),
        logging.StreamHandler()
    ]
)

# Enable Azure SDK logging
logging.getLogger('azure').setLevel(logging.DEBUG)
```

## References

### Official Documentation

1. Microsoft. (2026). "Azure AI Foundry documentation." https://learn.microsoft.com/azure/ai-foundry/

2. Microsoft. (2026). "Deploy hosted agents in Azure AI Foundry." https://learn.microsoft.com/azure/ai-foundry/deploy-hosted-agents-quickstart

3. Microsoft. (2026). "Azure OpenAI Service documentation." https://learn.microsoft.com/azure/cognitive-services/openai/

4. Microsoft. (2026). "Azure Container Apps documentation." https://learn.microsoft.com/azure/container-apps/

5. LangChain. (2026). "Microsoft Foundry Tools (formerly Azure AI Services) tools integration" https://docs.langchain.com/oss/python/integrations/tools/azure_ai_services

### Tutorials and Learning Paths

6. Microsoft Learn. (2026). "Build and deploy Azure AI agents." https://learn.microsoft.com/training/paths/build-azure-ai-agents/

7. Microsoft. (2026). "Azure Agent Factory: Revolutionizing AI Agent Development." https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/azure-agent-factory

8. Microsoft Learn. (2026). "Get started with LangChain and LangGraph with Foundry" https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/langchain

9. Microsoft Learn. (2026). "Use Foundry Agent Service with LangGraph" https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/langchain-agents

10. Microsoft Learn. (2026). "Retrieval augmented generation (RAG) and indexes" https://learn.microsoft.com/en-us/azure/foundry/concepts/retrieval-augmented-generation

11. Microsoft Learn. (2026). "Develop applications with LlamaIndex and Microsoft Foundry" https://learn.microsoft.com/en-us/azure/foundry-classic/how-to/develop/llama-index

12. Microsoft Learn. (2026). "Retrieval-Augmented Generation (RAG) evaluators" https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/rag-evaluators

13. Microsoft Learn. (2026). "RAG chunking phase" https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase

### Sample Code

14. Microsoft. (2026). "Azure AI Agent Samples." https://github.com/Azure-Samples/azure-ai-agent-samples

15. Microsoft. (2026). "Semantic Kernel Examples." https://github.com/microsoft/semantic-kernel

### Best Practices

16. Microsoft. (2026). "Enterprise-Scale AI Agent Architecture." https://learn.microsoft.com/azure/architecture/ai/agent-architecture

17. Microsoft. (2026). "Security Best Practices for Azure AI." https://learn.microsoft.com/security/ai-agent-security

---

*Last Updated: June 2026*
