# Google Cloud

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Virtual Environment Setup](#virtual-environment-setup)
4. [Google Cloud Architecture Overview](#google-cloud-architecture-overview)
5. [Deployment Options](#deployment-options)
6. [Use Case: Customer Support Agent](#use-case-customer-support-agent)
7. [Step-by-Step Implementation](#step-by-step-implementation)
8. [Advanced Features](#advanced-features)
9. [Security and Governance](#security-and-governance)
10. [Monitoring and Observability](#monitoring-and-observability)
11. [Cost Optimization](#cost-optimization)
12. [Troubleshooting](#troubleshooting)
13. [References](#references)

## Introduction

Google Cloud Platform provides an infrastructure and services for deploying production-ready AI agents. The platform combines managed Model Context Protocol (MCP) servers, the Agent Development Kit (ADK), and serverless deployment options to create scalable, secure, and observable agent systems.

### Key Components

**Vertex AI Agent Engine**: A fully managed service for deploying and running AI agents with automatic scaling, session management, and built-in memory.

**Cloud Run**: A serverless container platform that enables rapid deployment of containerized agents with automatic scaling and pay-per-use pricing.

**Agent Development Kit (ADK)**: An open-source Python and Java framework for building code-first agents with integrated tooling.

**Google-Managed MCP Servers**: Pre-built, production-ready connectors for Google services (Maps, BigQuery, GKE, Cloud Run) that require no infrastructure provisioning.

### Why Choose Google Cloud for AI Agents

1. **Production-Ready Infrastructure**: Automatic scaling, session isolation, and zero-downtime deployments
2. **Enterprise Security**: Native integration with Cloud IAM, VPC Service Controls, and Model Armor
3. **Unified Observability**: Cloud Audit Logs provide centralized visibility across all agent operations
4. **Rapid Development**: ADK framework accelerates development with built-in patterns for common agent architectures
5. **Ecosystem Integration**: Seamless connection to Google Workspace, BigQuery, and other Google services

## Prerequisites

Before beginning deployment, ensure you have:

### Required Accounts and Access

- A Google Cloud account with billing enabled
- Appropriate IAM permissions (Vertex AI User, Cloud Run Admin, or Project Editor)
- A project in Google Cloud Console

### Required Software

- Python 3.10 or later
- gcloud CLI installed and configured
- Docker Desktop (for local testing)
- Git (for version control)

### API Enablement

Enable the following APIs in your Google Cloud project:

```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudresourcemanager.googleapis.com
gcloud services enable iam.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### Authentication

Authenticate with Google Cloud:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login
```

## Virtual Environment Setup

Creating a virtual environment ensures dependency isolation and prevents conflicts with system packages.

### Initial Setup

1. **Create a project directory**:

```bash
mkdir gcp-ai-agent
cd gcp-ai-agent
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
# Google Cloud SDKs
google-cloud-aiplatform>=1.38.0
google-cloud-logging>=3.5.0
google-cloud-storage>=2.10.0

# ADK Framework
adk-python>=0.1.0

# Agent dependencies
pydantic>=2.5.0
python-dotenv>=1.0.0
fastapi>=0.104.0
uvicorn>=0.24.0

# MCP integration
mcp>=0.1.0

# Observability
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-gcp-trace>=1.6.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```bash
# Google Cloud Configuration
PROJECT_ID=your-project-id
LOCATION=us-central1
VERTEX_AI_ENDPOINT=us-central1-aiplatform.googleapis.com

# Model Configuration
MODEL_ID=gemini-2.0-flash
TEMPERATURE=0.7
MAX_TOKENS=2048

# MCP Configuration
MAPS_API_KEY=your-maps-api-key

# Application Configuration
PORT=8080
LOG_LEVEL=INFO
```

### VS Code Integration

For VS Code users, create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black"
}
```

## Google Cloud Architecture Overview

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Client Layer                        │
│  (Web App, Mobile App, API Client, Command Line)           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Cloud Run / Agent Engine                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              AI Agent (ADK Application)               │  │
│  │  ┌─────────────────────────────────────┐             │  │
│  │  │    Orchestration Layer              │             │  │
│  │  │  - Request routing                  │             │  │
│  │  │  - Tool selection                   │             │  │
│  │  │  - Memory management                │             │  │
│  │  │  - Response synthesis               │             │  │
│  │  └─────────────────────────────────────┘             │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
          │              │               │              │
          ▼              ▼               ▼              ▼
┌──────────────┐ ┌────────────┐ ┌─────────────┐ ┌────────────┐
│  Gemini API  │ │Google MCP  │ │MemoryStore  │ │Cloud Audit │
│  (Vertex AI) │ │  Servers   │ │  (Redis)    │ │    Logs    │
└──────────────┘ └────────────┘ └─────────────┘ └────────────┘
                      │
                      ├─ Maps Grounding Lite
                      ├─ BigQuery MCP
                      ├─ Cloud Run MCP
                      └─ GKE MCP
```

### Data Flow

1. **Client Request**: User or application sends request to Cloud Run endpoint
2. **Agent Orchestration**: ADK agent processes request and determines required tools
3. **Tool Invocation**: Agent calls Google MCP servers or custom tools
4. **LLM Reasoning**: Gemini model processes context and generates response
5. **Memory Update**: Session state stored in Memorystore for continuity
6. **Response Delivery**: Structured response returned to client
7. **Audit Logging**: All operations logged to Cloud Audit Logs

## Deployment Options

### Option 1: Managed Deployment with Vertex AI Agent Engine

**Best for**: Production workloads requiring minimal operational overhead

**Advantages**:
- Fully managed infrastructure
- Automatic scaling and session management
- Built-in authentication and authorization
- Integrated monitoring and logging

**Deployment**:

```bash
# Using ADK CLI
adk deploy agent_engine \
    --project $PROJECT_ID \
    --region $LOCATION \
    support_agent
```

### Option 2: Serverless Deployment with Cloud Run

**Best for**: Container-based deployments with custom configurations

**Advantages**:
- Full control over container environment
- Multi-region deployment support
- Custom domain mapping
- WebSocket and HTTP/2 support

**Deployment**:

```bash
# Deploy from source
gcloud run deploy support-agent \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars PROJECT_ID=$PROJECT_ID,MODEL_ID=gemini-2.0-flash
```

### Option 3: Hybrid Deployment

**Best for**: Complex multi-agent systems with diverse requirements

Combine managed Agent Engine for stateful agents with Cloud Run for stateless components.

## Use Case: Customer Support Agent

### Overview

A customer support agent that can retrieve order statuses from databases, search knowledge bases, and escalate complex issues to humans demonstrates multi-tool orchestration with Gemini models.

### Architecture

```
User Query → Support Agent → [get_order_status, search_knowledge_base, escalate_to_human]
                    ↓
              Gemini 2.0 Flash
                    ↓
          Response with Tool Calls
                    ↓
          Execute Tools → Return Results
                    ↓
          Final Response to User
```

### Sample Implementation

```python
from google.adk.agents import Agent
import datetime

# Define a tool for the agent
def get_order_status(order_id: str) -> dict:
    """Retrieves order status information from the database.
    
    Args:
        order_id: The unique identifier for the order
        
    Returns:
        Dictionary containing order status and delivery information
    """
    # In production, this would query your database
    return {
        "order_id": order_id,
        "status": "shipped",
        "delivery_date": "2026-04-25",
        "tracking_number": "1Z999AA10123456784"
    }

def search_knowledge_base(query: str) -> str:
    """Searches the company knowledge base for product information.
    
    Args:
        query: The search query
        
    Returns:
        Relevant information from the knowledge base
    """
    # In production, this would search your documentation
    return f"Found relevant information for: {query}"

def escalate_to_human(reason: str, customer_id: str) -> dict:
    """Creates a support ticket for human review.
    
    Args:
        reason: Explanation of why escalation is needed
        customer_id: The customer's unique identifier
        
    Returns:
        Ticket information
    """
    return {
        "ticket_id": f"TICKET-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "status": "pending_review",
        "assigned_to": "support_team"
    }

# Initialize the agent
support_agent = Agent(
    name="support_agent",
    model="gemini-2.0-flash",
    instruction="""You are a helpful customer support assistant. 
    
    Your capabilities:
    - Check order statuses by order ID
    - Search the knowledge base for product information
    - Escalate complex issues to human support agents
    
    Guidelines:
    - Always be polite and professional
    - Verify order IDs before checking status
    - Search knowledge base before escalating
    - Provide clear, concise responses
    - When escalating, explain the reason clearly""",
    tools=[get_order_status, search_knowledge_base, escalate_to_human]
)
```

## Step-by-Step Implementation

### Step 1: Initialize Project Structure

```bash
# Create project directory
mkdir customer-support-agent
cd customer-support-agent

# Create directory structure
mkdir -p src/{agents,tools,config}
mkdir tests
```

### Step 2: Create Agent Configuration

Create `src/config/agent_config.py`:

```python
"""Agent configuration settings."""
import os
from dataclasses import dataclass

@dataclass
class AgentConfig:
    """Configuration for the customer support agent."""
    
    project_id: str
    location: str
    model_id: str
    temperature: float
    max_tokens: int
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            project_id=os.getenv("PROJECT_ID"),
            location=os.getenv("LOCATION", "us-central1"),
            model_id=os.getenv("MODEL_ID", "gemini-2.0-flash"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2048"))
        )
```

### Step 3: Implement Tools

Create `src/tools/support_tools.py`:

```python
"""Support agent tools for order management and knowledge base access."""
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SupportTools:
    """Collection of tools for customer support agent."""
    
    def __init__(self, database_client, knowledge_base_client):
        """Initialize support tools with required clients.
        
        Args:
            database_client: Client for database operations
            knowledge_base_client: Client for knowledge base searches
        """
        self.database = database_client
        self.kb = knowledge_base_client
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Retrieve order status from database.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Order status information
        """
        try:
            order = self.database.get_order(order_id)
            return {
                "order_id": order_id,
                "status": order.status,
                "delivery_date": order.delivery_date,
                "items": order.items
            }
        except Exception as e:
            logger.error(f"Error retrieving order {order_id}: {e}")
            return {"error": "Order not found"}
    
    def search_knowledge_base(self, query: str) -> str:
        """Search knowledge base for relevant information.
        
        Args:
            query: Search query
            
        Returns:
            Relevant information from knowledge base
        """
        try:
            results = self.kb.search(query, top_k=3)
            return "\n\n".join([r.content for r in results])
        except Exception as e:
            logger.error(f"Knowledge base search error: {e}")
            return "Unable to search knowledge base at this time"
```

### Step 4: Create Main Agent Application

Create `src/agents/support_agent.py`:

```python
"""Customer support agent implementation."""
from google.adk.agents import Agent
from google.adk.tools import McpToolset
from google.adk.tools.streamable_http import StreamableHTTPConnectionParams
import os

class CustomerSupportAgent:
    """Customer support AI agent with order management capabilities."""
    
    def __init__(self, config):
        """Initialize the customer support agent.
        
        Args:
            config: Agent configuration object
        """
        self.config = config
        self.agent = self._build_agent()
    
    def _build_agent(self) -> Agent:
        """Build the agent with tools and configuration."""
        
        # Initialize MCP tools for Google services if needed
        maps_tools = McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url="https://mapstools.googleapis.com/mcp",
                headers={"X-Goog-Api-Key": os.environ.get("MAPS_API_KEY")}
            )
        )
        
        return Agent(
            name="customer_support",
            model=self.config.model_id,
            instruction=self._get_system_prompt(),
            tools=[
                self._get_order_status,
                self._search_knowledge_base,
                self._escalate_to_human,
                maps_tools
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
    
    def _get_system_prompt(self) -> str:
        """Return the agent's system prompt."""
        return """You are an expert customer support agent for our e-commerce platform.
        
        Your responsibilities:
        1. Help customers track their orders
        2. Answer product questions using the knowledge base
        3. Escalate complex issues to human agents when necessary
        
        Best practices:
        - Always verify order IDs before looking them up
        - Search the knowledge base before escalating
        - Be empathetic and professional
        - Provide clear, actionable information
        - Set appropriate expectations for resolution times
        """
    
    def _get_order_status(self, order_id: str) -> dict:
        """Tool function to retrieve order status."""
        # Implementation here
        pass
    
    def _search_knowledge_base(self, query: str) -> str:
        """Tool function to search knowledge base."""
        # Implementation here
        pass
    
    def _escalate_to_human(self, reason: str, customer_id: str) -> dict:
        """Tool function to create escalation ticket."""
        # Implementation here
        pass
    
    def run(self, user_message: str) -> str:
        """Process a user message and return agent response.
        
        Args:
            user_message: The user's input message
            
        Returns:
            Agent's response
        """
        result = self.agent.run(user_message)
        return result.message
```

### Step 5: Deploy to Cloud Run

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY main.py .

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run the application
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
```

Create `main.py`:

```python
"""FastAPI application for customer support agent."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agents.support_agent import CustomerSupportAgent
from src.config.agent_config import AgentConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Customer Support Agent API")

# Initialize agent
config = AgentConfig.from_env()
agent = CustomerSupportAgent(config)

class MessageRequest(BaseModel):
    """Request model for chat messages."""
    message: str
    session_id: str = None

class MessageResponse(BaseModel):
    """Response model for chat messages."""
    response: str
    session_id: str

@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """Process a chat message through the support agent.
    
    Args:
        request: The chat message request
        
    Returns:
        Agent's response
    """
    try:
        logger.info(f"Processing message: {request.message}")
        response = agent.run(request.message)
        return MessageResponse(
            response=response,
            session_id=request.session_id or "default"
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
```

Deploy to Cloud Run:

```bash
# Build and deploy
gcloud run deploy customer-support-agent \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars PROJECT_ID=$PROJECT_ID
```

### Step 6: Alternative Deployment to Agent Engine

Using ADK CLI:

```bash
# Create deployment configuration
adk config create agent_engine \
    --project $PROJECT_ID \
    --region us-central1 \
    --agent-name customer-support

# Deploy agent
adk deploy agent_engine support_agent
```

## Advanced Features

### Memory Management

Implement conversation memory for context retention:

```python
from langgraph.checkpoint.memory import MemorySaver

# Initialize memory store
memory = MemorySaver()

# Create agent with memory
agent = Agent(
    name="support_agent",
    model="gemini-2.0-flash",
    checkpointer=memory,
    # ... other configuration
)

# Use with session management
config = {"configurable": {"thread_id": "customer-12345"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Where's my order?"}]},
    config=config
)
```

### Structured Output

Define response schemas for consistent output:

```python
from pydantic import BaseModel
from typing import Optional

class SupportResponse(BaseModel):
    """Structured response from support agent."""
    message: str
    order_info: Optional[dict] = None
    escalated: bool = False
    ticket_id: Optional[str] = None

# Configure agent with structured output
agent = Agent(
    name="support_agent",
    model="gemini-2.0-flash",
    response_format=SupportResponse,
    # ... other configuration
)
```

### Multi-Agent Orchestration

Create specialized agents for different tasks:

```python
# Create specialized agents
order_agent = Agent(
    name="order_specialist",
    tools=[get_order_status, update_order],
    instruction="You specialize in order management"
)

product_agent = Agent(
    name="product_specialist",
    tools=[search_products, get_product_details],
    instruction="You specialize in product information"
)

# Create coordinator agent
coordinator = Agent(
    name="coordinator",
    tools=[order_agent.as_tool(), product_agent.as_tool()],
    instruction="Route requests to appropriate specialist"
)
```

## Security and Governance

### IAM Configuration

Create service account for agent:

```bash
# Create service account
gcloud iam service-accounts create agent-service-account \
    --display-name="Customer Support Agent"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:agent-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:agent-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"
```

### VPC Service Controls

Enable VPC-SC for data protection:

```bash
# Create service perimeter
gcloud access-context-manager perimeters create agent-perimeter \
    --title="Agent Service Perimeter" \
    --resources=projects/$PROJECT_NUMBER \
    --restricted-services=aiplatform.googleapis.com \
    --policy=$POLICY_ID
```

### Model Armor Integration

Enable content filtering:

```bash
# Configure Model Armor
gcloud model-armor floorsettings update \
    --full-uri="projects/$PROJECT_ID/locations/global/floorSetting" \
    --enable-floor-setting-enforcement=TRUE \
    --add-integrated-services=GOOGLE_MCP_SERVER \
    --enable-google-mcp-server-cloud-logging \
    --malicious-uri-filter-settings-enforcement=ENABLED
```

## Monitoring and Observability

### Cloud Logging

Query agent logs:

```bash
# View agent logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=customer-support-agent" \
    --limit 50 \
    --format json
```

### Cloud Trace

Enable distributed tracing:

```python
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
tracer_provider = TracerProvider()
cloud_trace_exporter = CloudTraceSpanExporter()
tracer_provider.add_span_processor(
    BatchSpanProcessor(cloud_trace_exporter)
)
trace.set_tracer_provider(tracer_provider)
```

### Custom Metrics

Track agent performance:

```python
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{PROJECT_ID}"

# Create custom metric
series = monitoring_v3.TimeSeries()
series.metric.type = "custom.googleapis.com/agent/response_time"
series.resource.type = "cloud_run_revision"

# Write metric datapoint
client.create_time_series(name=project_name, time_series=[series])
```

## Cost Optimization

### Model Selection

Choose appropriate model tier:

| Model | Use Case | Cost/1M tokens |
|-------|----------|----------------|
| gemini-2.0-flash | General support, high volume | Low |
| gemini-pro | Complex reasoning | Medium |
| gemini-ultra | Critical decisions | High |

### Request Batching

Implement request batching for efficiency:

```python
from typing import List

async def process_batch(messages: List[str]) -> List[str]:
    """Process multiple messages in batch."""
    tasks = [agent.arun(msg) for msg in messages]
    return await asyncio.gather(*tasks)
```

### Token Management

Monitor and optimize token usage:

```python
def estimate_tokens(text: str) -> int:
    """Estimate token count for input text."""
    return len(text) // 4  # Rough estimate

def truncate_context(messages: List[dict], max_tokens: int) -> List[dict]:
    """Truncate conversation history to fit token limit."""
    total_tokens = sum(estimate_tokens(m["content"]) for m in messages)
    
    while total_tokens > max_tokens and len(messages) > 1:
        messages.pop(0)
        total_tokens = sum(estimate_tokens(m["content"]) for m in messages)
    
    return messages
```

## Troubleshooting

### Common Issues

**Issue**: Agent not finding tools

**Solution**:
```python
# Verify tool registration
print(agent.available_tools)

# Check tool signatures
import inspect
print(inspect.signature(get_order_status))
```

**Issue**: Authentication errors

**Solution**:
```bash
# Refresh application default credentials
gcloud auth application-default login

# Verify service account permissions
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:agent-service-account@*"
```

**Issue**: Slow response times

**Solution**:
- Enable connection pooling
- Implement caching for frequent queries
- Use async/await for concurrent operations

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable ADK debug mode
os.environ["ADK_DEBUG"] = "true"
```

### Performance Profiling

Profile agent execution:

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run agent
response = agent.run("Check order status")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## References

### Official Documentation

1. Google Cloud. (2026). "Host AI agents on Cloud Run." https://docs.cloud.google.com/run/docs/ai-agents

2. Google Cloud. (2026). "Deploy an agent - Agent Builder Documentation." https://docs.cloud.google.com/agent-builder/agent-engine/deploy

3. Google Cloud. (2026). "ADK Python Documentation." https://google.github.io/adk-docs/

### Tutorials and Codelabs

4. Google Cloud. (2026). "Local to Cloud: Full-Stack App Migration with Gemini CLI and Cloud SQL MCP." https://codelabs.developers.google.com/ai-mcp-dk-csql

5. Google Cloud. (2026). "Build Multi-Agent Systems with ADK." https://codelabs.developers.google.com/codelabs/production-ready-ai-with-gc/3-developing-agents/build-a-multi-agent-system-with-adk

### Sample Applications

6. Strebel, D. (2026). "ADK Cityscape - AI Agent Example." https://github.com/danistrebel/adk-cityscape

7. Google Cloud. (2026). "ADK Sample Applications." https://github.com/google/adk-samples

### Best Practices

8. Google Cloud. (2026). "A developer's guide to production-ready AI agents." https://cloud.google.com/blog/products/ai-machine-learning/a-devs-guide-to-production-ready-ai-agents

9. Google Cloud. (2026). "Build Your First ADK Agent Workforce." https://cloud.google.com/blog/topics/developers-practitioners/build-your-first-adk-agent-workforce

---

*Last Updated: April 2026*
