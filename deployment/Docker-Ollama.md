# Docker + Ollama

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Virtual Environment Setup](#virtual-environment-setup)
4. [Architecture Overview](#architecture-overview)
5. [Deployment Workflow](#deployment-workflow)
6. [Use Case: Document Analysis Agent](#use-case-document-analysis-agent)
7. [Step-by-Step Implementation](#step-by-step-implementation)
8. [Advanced Features](#advanced-features)
9. [Security and Privacy](#security-and-privacy)
10. [Performance Optimization](#performance-optimization)
11. [Troubleshooting](#troubleshooting)
12. [References](#references)

## Introduction

Deploying AI agents on local infrastructure using Docker and Ollama provides organizations with complete control over data privacy, infrastructure costs, and model customization. This approach is particularly valuable for:

- **Privacy-Sensitive Applications**: Healthcare, legal, financial services requiring on-premises data processing
- **Development and Testing**: Rapid prototyping without cloud costs or API rate limits
- **Air-Gapped Environments**: Secure facilities with no internet connectivity
- **Cost Optimization**: Eliminate per-token API fees for high-volume workloads

### Key Components

**Docker**: Containerization platform ensuring consistent runtime environments across development, testing, and production.

**Ollama**: Local LLM runtime supporting models like Llama 3, Mistral, CodeLlama, and Qwen with optimized inference on CPU and GPU.

**LangChain**: Framework for building LLM applications with chains, agents, and memory management.

**Langflow**: Visual development environment for creating agent workflows with drag-and-drop components.

**CrewAI**: Multi-agent orchestration framework enabling teams of specialized agents to collaborate on complex tasks.

### Why Choose Local Deployment

1. **Data Sovereignty**: All data processing occurs on-premises, never transmitted to external servers
2. **Zero API Costs**: Eliminate per-token fees; costs limited to hardware and electricity
3. **Unlimited Usage**: No rate limits, quotas, or usage caps
4. **Model Customization**: Fine-tune models on proprietary data without vendor restrictions
5. **Offline Operation**: Full functionality without internet connectivity

### Supported Models (Ollama)

| Model | Parameters | RAM Required | Best For |
|-------|------------|--------------|----------|
| llama3:8b | 8 billion | 8 GB | General purpose, fast inference |
| llama3:70b | 70 billion | 48 GB | Complex reasoning, high quality |
| mistral:7b | 7 billion | 8 GB | Instruction following, coding |
| codellama:34b | 34 billion | 20 GB | Code generation, debugging |
| qwen2.5:32b | 32 billion | 20 GB | Multilingual, mathematics |
| mixtral:8x7b | 46.7 billion | 32 GB | Mixture of experts, balanced |

## Prerequisites

### Hardware Requirements

**Minimum Configuration**:
- CPU: 4 cores (Intel i5/AMD Ryzen 5 or better)
- RAM: 16 GB
- Storage: 50 GB free space (SSD recommended)
- GPU: Optional (NVIDIA with CUDA support significantly accelerates inference)

**Recommended Configuration**:
- CPU: 8+ cores (Intel i7/i9 or AMD Ryzen 7/9)
- RAM: 32 GB (64 GB for larger models)
- Storage: 100 GB+ NVMe SSD
- GPU: NVIDIA RTX 3060+ with 12+ GB VRAM

### Software Requirements

- **Operating System**: Linux (Ubuntu 22.04+), macOS (12+), or Windows 11 with WSL2
- **Docker Desktop**: v24.0+ or Docker Engine v24.0+
- **Docker Compose**: v2.20+
- **Python**: 3.10 or later
- **Git**: For version control
- **NVIDIA Container Toolkit**: If using GPU acceleration

### Installation

**Docker Installation (Ubuntu)**:

```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add user to docker group (avoid sudo)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

**Ollama Installation**:

```bash
# Linux / macOS installation
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version

# Windows: Download installer from https://ollama.com/download
```

**NVIDIA GPU Support (Optional but Recommended)**:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

## Virtual Environment Setup

### Initial Setup

1. **Create project directory**:

```bash
mkdir document-analysis-agent
cd document-analysis-agent
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

Create `requirements.txt`:

```txt
# LLM frameworks
langchain>=0.1.0
langchain-community>=0.0.20
ollama>=0.1.6

# Agent frameworks
crewai>=0.1.0
langgraph>=0.0.55

# Document processing
pypdf>=3.17.0
python-docx>=1.1.0
python-pptx>=0.6.23
openpyxl>=3.1.2
pillow>=10.1.0
pytesseract>=0.3.10  # OCR support

# Vector storage
chromadb>=0.4.22
faiss-cpu>=1.7.4  # or faiss-gpu for GPU support

# Web framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
streamlit>=1.29.0  # Optional: for web UI

# Utilities
pydantic>=2.5.0
python-dotenv>=1.0.0
python-multipart>=0.0.6
aiofiles>=23.2.1

# Observability
prometheus-client>=0.19.0

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

Create `.env` file:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:32b
OLLAMA_TEMPERATURE=0.7
OLLAMA_NUM_CTX=4096  # Context window size

# Vector Database
VECTOR_DB_PATH=./data/vector_store
VECTOR_DB_COLLECTION=documents

# Application Configuration
APP_NAME=Document Analysis Agent
APP_PORT=8000
UPLOAD_DIR=./uploads
MAX_FILE_SIZE_MB=50

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/agent.log
```

## Architecture Overview

### Component Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Client Interface                        │
│     (Web UI, API, CLI, Jupyter Notebook)                 │
└──────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│               FastAPI Application Server                  │
│                  (Port 8000)                             │
└──────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┴──────────────┐
        ▼                              ▼
┌─────────────────┐          ┌──────────────────┐
│  Document       │          │   Query          │
│  Processing     │          │   Processing     │
│  Pipeline       │          │   Pipeline       │
└─────────────────┘          └──────────────────┘
        │                              │
        ▼                              ▼
┌──────────────────────────────────────────────────────────┐
│              Document Analysis Agent                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │             LangChain Agent                        │  │
│  │  - Document summarization                          │  │
│  │  - Question answering                              │  │
│  │  - Entity extraction                               │  │
│  │  - Semantic search                                 │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
        │              │              │
        ▼              ▼              ▼
┌──────────┐  ┌──────────────┐  ┌──────────────┐
│  Ollama  │  │   ChromaDB   │  │  Document    │
│  Server  │  │   Vector     │  │  Storage     │
│          │  │   Store      │  │              │
└──────────┘  └──────────────┘  └──────────────┘
     │
     ├─ qwen2.5:32b (primary model)
     ├─ llama3:8b (fast queries)
     └─ nomic-embed-text (embeddings)
```

### Docker Compose Architecture

```yaml
# All services run in isolated containers
┌─────────────────────────────────────────┐
│  Docker Network: agent-network          │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  ollama-service                   │  │
│  │  - Runs Ollama server             │  │
│  │  - GPU access (if available)      │  │
│  │  - Volume: ./ollama-data          │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  agent-service                    │  │
│  │  - Python application             │  │
│  │  - FastAPI server                 │  │
│  │  - Volumes: ./src, ./data         │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  chromadb-service                 │  │
│  │  - Vector database                │  │
│  │  - Persistent storage             │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Deployment Workflow

### Step-by-Step Deployment

1. **Pull Ollama models**
2. **Build Docker images**
3. **Start services with Docker Compose**
4. **Initialize vector database**
5. **Upload and process documents**
6. **Query the agent**

### Quick Start

```bash
# Clone or create project
git clone <your-repo> document-agent
cd document-agent

# Pull required Ollama models
ollama pull qwen2.5:32b
ollama pull nomic-embed-text

# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Access application
open http://localhost:8000
```

## Use Case: Document Analysis Agent

### Overview

A document analysis agent that can:
1. Ingest PDFs, Word documents, PowerPoints, and spreadsheets
2. Extract text, tables, and images
3. Answer questions about document content
4. Generate summaries and extract key insights
5. Perform semantic search across document collections
6. Compare and correlate information across multiple documents

### Sample Implementation

```python
"""Document analysis agent using Ollama and LangChain."""
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from pypdf import PdfReader
import os
from typing import List, Dict, Any

class DocumentAnalysisAgent:
    """Agent for analyzing and querying documents."""
    
    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "qwen2.5:32b",
        embedding_model: str = "nomic-embed-text"
    ):
        """Initialize the document analysis agent.
        
        Args:
            ollama_host: Ollama server URL
            model: Primary LLM model for reasoning
            embedding_model: Model for generating embeddings
        """
        # Initialize LLM
        self.llm = Ollama(
            base_url=ollama_host,
            model=model,
            temperature=0.7
        )
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            base_url=ollama_host,
            model=embedding_model
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name="documents",
            embedding_function=self.embeddings,
            persist_directory="./data/chroma"
        )
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_pdf(self, file_path: str) -> List[str]:
        """Extract text from PDF document.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of text chunks
        """
        reader = PdfReader(file_path)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text()
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def index_document(self, file_path: str, metadata: Dict[str, Any] = None) -> int:
        """Index a document into the vector store.
        
        Args:
            file_path: Path to document
            metadata: Additional metadata to store
            
        Returns:
            Number of chunks indexed
        """
        # Extract text (supports PDF for now)
        if file_path.endswith('.pdf'):
            chunks = self.load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata['source'] = file_path
        
        # Create documents with metadata
        metadatas = [metadata] * len(chunks)
        
        # Add to vector store
        self.vector_store.add_texts(
            texts=chunks,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question about indexed documents.
        
        Args:
            question: User question
            
        Returns:
            Answer with source citations
        """
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )
        
        # Get answer
        result = qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in result["source_documents"]
            ]
        }
    
    def summarize_document(self, file_path: str) -> str:
        """Generate a summary of a document.
        
        Args:
            file_path: Path to document
            
        Returns:
            Document summary
        """
        # Load document
        chunks = self.load_pdf(file_path)
        
        # Combine chunks (limit to prevent context overflow)
        full_text = "\n\n".join(chunks[:10])  # First 10 chunks
        
        # Create summary prompt
        prompt = f"""Please provide a summary of the following document.
Include the main topics, key findings, and important conclusions.

Document content:
{full_text}

Summary:"""
        
        # Generate summary
        summary = self.llm.invoke(prompt)
        return summary
    
    def extract_entities(self, file_path: str) -> List[Dict[str, str]]:
        """Extract named entities from document.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of extracted entities
        """
        chunks = self.load_pdf(file_path)
        sample_text = "\n\n".join(chunks[:5])
        
        prompt = f"""Extract all named entities from the following text. 
Categorize them as PERSON, ORGANIZATION, LOCATION, DATE, or OTHER.

Text:
{sample_text}

Format your response as:
- Entity Name (Category)

Entities:"""
        
        response = self.llm.invoke(prompt)
        
        # Parse response (simplified)
        entities = []
        for line in response.split('\n'):
            if line.strip().startswith('-'):
                entities.append({"text": line.strip('- '), "raw": line})
        
        return entities
```

## Step-by-Step Implementation

### Step 1: Create Project Structure

```bash
# Create directory structure
mkdir -p document-agent/{src,data,uploads,logs,tests}
cd document-agent

# Create subdirectories
mkdir -p src/{agents,tools,api,utils}
mkdir -p data/{chroma,ollama}
```

### Step 2: Create Docker Compose Configuration

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-server
    ports:
      - "11434:11434"
    volumes:
      - ./data/ollama:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    networks:
      - agent-network
    restart: unless-stopped

  chromadb:
    image: chromadb/chroma:latest
    container_name: chromadb-server
    ports:
      - "8001:8000"
    volumes:
      - ./data/chroma:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE
    networks:
      - agent-network
    restart: unless-stopped

  agent-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: document-agent-api
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src
      - ./uploads:/app/uploads
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - OLLAMA_MODEL=qwen2.5:32b
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
    depends_on:
      - ollama
      - chromadb
    networks:
      - agent-network
    restart: unless-stopped

networks:
  agent-network:
    driver: bridge

volumes:
  ollama-data:
  chroma-data:
```

### Step 3: Create Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p uploads data logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### Step 4: Implement FastAPI Application

Create `src/api/main.py`:

```python
"""FastAPI application for document analysis agent."""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from pathlib import Path
import logging

from src.agents.document_agent import DocumentAnalysisAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Analysis Agent",
    description="AI agent for document analysis using local LLMs",
    version="1.0.0"
)

# Initialize agent
agent = DocumentAnalysisAgent(
    ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    model=os.getenv("OLLAMA_MODEL", "qwen2.5:32b")
)

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class QuestionRequest(BaseModel):
    """Request model for questions."""
    question: str

class QuestionResponse(BaseModel):
    """Response model for answers."""
    answer: str
    sources: List[dict]

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "document-agent"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document.
    
    Args:
        file: Uploaded file (PDF, DOCX, etc.)
        
    Returns:
        Success message with chunk count
    """
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing file: {file.filename}")
        
        # Index document
        chunk_count = agent.index_document(
            str(file_path),
            metadata={"filename": file.filename}
        )
        
        return {
            "message": "Document uploaded and indexed successfully",
            "filename": file.filename,
            "chunks": chunk_count
        }
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about indexed documents.
    
    Args:
        request: Question request
        
    Returns:
        Answer with source citations
    """
    try:
        logger.info(f"Processing question: {request.question}")
        
        result = agent.answer_question(request.question)
        
        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents(request: QuestionRequest, k: int = 5):
    """Search for relevant document chunks.
    
    Args:
        request: Search query
        k: Number of results
        
    Returns:
        Relevant document chunks
    """
    try:
        results = agent.search_documents(request.question, k=k)
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_document(filename: str):
    """Generate summary of a document.
    
    Args:
        filename: Name of uploaded file
        
    Returns:
        Document summary
    """
    try:
        file_path = UPLOAD_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        summary = agent.summarize_document(str(file_path))
        
        return {
            "filename": filename,
            "summary": summary
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    return {"documents": files, "count": len(files)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 5: Create Startup Script

Create `start.sh`:

```bash
#!/bin/bash

echo "Starting Document Analysis Agent..."

# Pull required Ollama models
echo "Pulling Ollama models..."
ollama pull qwen2.5:32b
ollama pull nomic-embed-text

# Start Docker Compose services
echo "Starting Docker services..."
docker compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Check service health
echo "Checking service health..."
curl -f http://localhost:8000/health || echo "Agent API not ready yet"
curl -f http://localhost:11434/api/tags || echo "Ollama not ready yet"

echo "Services started!"
echo "Agent API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Ollama: http://localhost:11434"

# Show logs
docker compose logs -f
```

Make executable and run:

```bash
chmod +x start.sh
./start.sh
```

### Step 6: Test the Agent

Create `test_agent.py`:

```python
"""Test script for document analysis agent."""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health check:", response.json())

def test_upload(file_path: str):
    """Test document upload."""
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    print("Upload result:", response.json())

def test_question(question: str):
    """Test question answering."""
    data = {"question": question}
    response = requests.post(f"{BASE_URL}/ask", json=data)
    result = response.json()
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources:")
    for source in result['sources']:
        print(f"  - {source['source']}: {source['content'][:100]}...")

if __name__ == "__main__":
    test_health()
    
    # Upload a document
    # test_upload("sample.pdf")
    
    # Ask questions
    test_question("What are the main topics in the document?")
    test_question("Summarize the key findings.")
```

Run test:

```bash
python test_agent.py
```

## Advanced Features

### Multi-Agent System with CrewAI

Create specialized agents for different tasks:

```python
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama

# Initialize LLM
llm = Ollama(model="qwen2.5:32b")

# Define specialized agents
researcher = Agent(
    role="Research Analyst",
    goal="Extract and analyze information from documents",
    backstory="Expert at finding relevant information and insights",
    llm=llm,
    verbose=True
)

summarizer = Agent(
    role="Content Summarizer",
    goal="Create concise, accurate summaries",
    backstory="Skilled at distilling complex information into clear summaries",
    llm=llm,
    verbose=True
)

critic = Agent(
    role="Quality Reviewer",
    goal="Verify accuracy and completeness of analysis",
    backstory="Detail-oriented reviewer ensuring high quality output",
    llm=llm,
    verbose=True
)

# Define tasks
research_task = Task(
    description="Analyze the document and extract key information",
    agent=researcher
)

summary_task = Task(
    description="Create a summary of the findings",
    agent=summarizer
)

review_task = Task(
    description="Review the summary for accuracy and completeness",
    agent=critic
)

# Create crew
document_crew = Crew(
    agents=[researcher, summarizer, critic],
    tasks=[research_task, summary_task, review_task],
    verbose=True
)

# Execute workflow
result = document_crew.kickoff()
print(result)
```

### Visual Workflow with Langflow

Install Langflow:

```bash
pip install langflow

# Start Langflow server
langflow run
```

Access at `http://localhost:7860` and create visual agent workflows.

### Advanced Document Processing

```python
from docx import Document as DocxDocument
from pptx import Presentation
import pandas as pd

class AdvancedDocumentProcessor:
    """Process multiple document formats."""
    
    @staticmethod
    def extract_from_word(file_path: str) -> str:
        """Extract text from Word document."""
        doc = DocxDocument(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    @staticmethod
    def extract_from_powerpoint(file_path: str) -> str:
        """Extract text from PowerPoint."""
        prs = Presentation(file_path)
        text = []
        
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
        
        return "\n".join(text)
    
    @staticmethod
    def extract_from_excel(file_path: str) -> str:
        """Extract data from Excel."""
        df = pd.read_excel(file_path)
        return df.to_string()
```

## Security and Privacy

### Data Isolation

All processing occurs locally:

```bash
# Verify no external network calls
docker compose exec agent-api tcpdump -i any port 443 -c 10

# Should show no HTTPS traffic to cloud services
```

### Encryption at Rest

Encrypt stored documents:

```python
from cryptography.fernet import Fernet
import os

class EncryptionManager:
    """Encrypt sensitive documents."""
    
    def __init__(self):
        key = os.getenv("ENCRYPTION_KEY") or Fernet.generate_key()
        self.cipher = Fernet(key)
    
    def encrypt_file(self, file_path: str) -> str:
        """Encrypt a file."""
        with open(file_path, "rb") as f:
            data = f.read()
        
        encrypted = self.cipher.encrypt(data)
        encrypted_path = f"{file_path}.encrypted"
        
        with open(encrypted_path, "wb") as f:
            f.write(encrypted)
        
        return encrypted_path
    
    def decrypt_file(self, encrypted_path: str) -> bytes:
        """Decrypt a file."""
        with open(encrypted_path, "rb") as f:
            encrypted_data = f.read()
        
        return self.cipher.decrypt(encrypted_data)
```

### Access Control

Implement authentication:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            os.getenv("JWT_SECRET"),
            algorithms=["HS256"]
        )
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )

@app.post("/ask", dependencies=[Depends(verify_token)])
async def ask_question(request: QuestionRequest):
    # Protected endpoint
    pass
```

## Performance Optimization

### Model Selection

Choose models based on hardware:

| Hardware | Recommended Model | Inference Speed |
|----------|------------------|-----------------|
| 8 GB RAM, CPU only | llama3:8b | ~20 tokens/sec |
| 16 GB RAM, CPU | mistral:7b | ~25 tokens/sec |
| 32 GB RAM, RTX 3060 | qwen2.5:32b | ~60 tokens/sec |
| 64 GB RAM, RTX 4090 | llama3:70b | ~40 tokens/sec |

### GPU Acceleration

Verify GPU usage:

```bash
# Monitor GPU while running inference
watch -n 1 nvidia-smi

# Expected output should show GPU utilization > 50%
```

### Batch Processing

Process multiple documents efficiently:

```python
async def batch_index_documents(file_paths: List[str]) -> Dict[str, int]:
    """Index multiple documents concurrently."""
    import asyncio
    
    async def index_one(path: str):
        return path, agent.index_document(path)
    
    tasks = [index_one(path) for path in file_paths]
    results = await asyncio.gather(*tasks)
    
    return dict(results)
```

### Caching

Implement response caching:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_embedding(text: str):
    """Cache embeddings for frequently queried text."""
    return agent.embeddings.embed_query(text)
```

## Troubleshooting

### Common Issues

**Issue**: Ollama models not downloading

**Solution**:
```bash
# Check Ollama service
docker compose logs ollama

# Pull models manually
docker compose exec ollama ollama pull qwen2.5:32b
```

**Issue**: Out of memory errors

**Solution**:
```bash
# Use smaller model
docker compose exec ollama ollama pull llama3:8b

# Or increase Docker memory limit in Docker Desktop settings
```

**Issue**: Slow inference times

**Solution**:
```python
# Reduce context window
llm = Ollama(
    model="qwen2.5:32b",
    num_ctx=2048,  # Reduced from 4096
    num_gpu=1  # Ensure GPU usage
)
```

**Issue**: ChromaDB connection errors

**Solution**:
```bash
# Restart ChromaDB
docker compose restart chromadb

# Check logs
docker compose logs chromadb
```

### Debug Mode

Enable verbose logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable LangChain debug
os.environ["LANGCHAIN_VERBOSE"] = "true"
```

## References

### Official Documentation

1. Ollama. (2026). "Ollama Documentation." https://ollama.com/docs

2. Docker. (2026). "Docker Compose Documentation." https://docs.docker.com/compose/

3. LangChain. (2026). "LangChain Documentation - Agents." https://python.langchain.com/docs/modules/agents/

4. ChromaDB. (2026). "Chroma Documentation." https://docs.trychroma.com/

### Tutorials and Guides

5. Ollama. (2026). "Quickstart Guide." https://github.com/ollama/ollama#quickstart

6. LangChain. (2026). "Building Local AI Agents." https://python.langchain.com/docs/use_cases/question_answering/

7. CrewAI. (2026). "Multi-Agent Systems Guide." https://docs.crewai.com/

### Sample Applications

8. Langflow. (2026). "Ollama Integration Examples." https://docs.langflow.org/integrations/ollama

9. Anthropic. (2026). "MCP Local Server Examples." https://github.com/modelcontextprotocol/servers

### Best Practices

10. Ollama Community. (2026). "Model Performance Benchmarks." https://ollama.com/blog/model-benchmarks

11. Docker. (2026). "Containerizing Python AI Applications." https://docs.docker.com/language/python/

---

*Last Updated: April 2026*
