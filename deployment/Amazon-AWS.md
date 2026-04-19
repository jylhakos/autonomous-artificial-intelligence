# Amazon Web Services

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Virtual Environment Setup](#virtual-environment-setup)
4. [AWS Architecture Overview](#aws-architecture-overview)
5. [Deployment Options](#deployment-options)
6. [Use Case: Mortgage Assistant Agent](#use-case-mortgage-assistant-agent)
7. [Step-by-Step Implementation](#step-by-step-implementation)
8. [Advanced Features](#advanced-features)
9. [Security and Governance](#security-and-governance)
10. [Monitoring and Observability](#monitoring-and-observability)
11. [Cost Optimization](#cost-optimization)
12. [Troubleshooting](#troubleshooting)
13. [References](#references)

## Introduction

Amazon Web Services provides an ecosystem for deploying AI agents with enterprise-grade security, scalability, and integration capabilities. AWS Bedrock AgentCore, combined with containerization on ECS/EKS and serverless compute on Lambda, enables organizations to build production-ready agentic systems that comply with strict regulatory requirements.

### Key Components

**Bedrock AgentCore**: A specialized runtime environment designed for hosting and orchestrating AI agents with built-in security controls and session management.

**Amazon Bedrock**: Fully managed service providing access to Claude 3.5 Sonnet, Titan, and other foundation models through a unified API.

**AWS Lambda**: Serverless compute platform for event-driven agent functions with automatic scaling and pay-per-invocation pricing.

**Amazon ECS/EKS**: Container orchestration services for deploying complex multi-agent systems with granular resource control.

**Amazon MemoryDB**: Redis-compatible in-memory database for agent conversation history and session state management.

### Why Choose AWS for AI Agents

1. **Enterprise Security**: Native IAM integration, VPC isolation, AWS PrivateLink for private connectivity
2. **Regulatory Compliance**: HIPAA, SOC 2, PCI DSS, GDPR-compliant infrastructure out of the box
3. **Hybrid Deployment**: Seamless integration between AWS cloud and on-premises environments via AWS Outposts
4. **Model Diversity**: Access to Claude, Titan, Llama, Mistral, and custom models through Bedrock
5. **Ecosystem Integration**: Direct connectivity to 200+ AWS services

## Prerequisites

Before beginning deployment, ensure you have:

### Required Accounts and Access

- An AWS account with administrative or PowerUser access
- AWS CLI v2 installed and configured
- Appropriate IAM permissions (Bedrock full access, Lambda execution, ECS/ECR access)

### Required Software

- Python 3.10 or later
- AWS CLI v2.x
- Docker Desktop (for containerization)
- Git (for version control)
- Node.js 18+ (for CDK deployments, optional)

### Service Activation

Enable Bedrock model access in your region:

```bash
# List available models in Bedrock
aws bedrock list-foundation-models --region us-east-1

# Request model access (one-time setup)
# Go to AWS Console → Bedrock → Model access → Request access
```

Configure AWS CLI:

```bash
# Configure AWS credentials
aws configure

# Set default region
export AWS_DEFAULT_REGION=us-east-1

# Verify access
aws sts get-caller-identity
```

### IAM Setup

Create necessary IAM policies:

```bash
# Create service role for Lambda
aws iam create-role \
    --role-name AgentLambdaRole \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }'

# Attach necessary policies
aws iam attach-role-policy \
    --role-name AgentLambdaRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
    --role-name AgentLambdaRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess
```

## Virtual Environment Setup

### Initial Setup

1. **Create a project directory**:

```bash
mkdir aws-mortgage-agent
cd aws-mortgage-agent
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
# AWS SDKs
boto3>=1.34.0
botocore>=1.34.0
aioboto3>=12.3.0

# Agent frameworks
langgraph>=0.0.55
langchain>=0.1.0
langchain-aws>=0.1.0
strands-agent-sdk>=0.1.0

# MCP integration
mcp>=0.1.0

# Web framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
mangum>=0.17.0  # For Lambda adapter

# Data processing
pydantic>=2.5.0
python-dotenv>=1.0.0

# Observability
aws-xray-sdk>=2.12.0
watchtower>=3.0.1  # CloudWatch integration

# Development tools
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
mypy>=1.8.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012

# Bedrock Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_RUNTIME_TIMEOUT=300
MAX_TOKENS=4096
TEMPERATURE=0.7

# AgentCore Configuration
AGENT_CORE_ENDPOINT=https://agentcore.bedrock.amazonaws.com
SESSION_TIMEOUT=3600

# Memory Configuration
MEMORYDB_ENDPOINT=your-memorydb-cluster.xxxxx.memorydb.amazonaws.com
MEMORYDB_PORT=6379

# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### Project Structure

Create the following directory structure:

```bash
mkdir -p src/{agents,tools,config,utils}
mkdir -p tests/{unit,integration}
mkdir infrastructure  # For IaC templates
mkdir docs
```

## AWS Architecture Overview

### Component Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Client Applications                       │
│     (Web App, Mobile, API Gateway, EventBridge)              │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     Application Load Balancer                 │
│                    (TLS termination, routing)                 │
└──────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌────────────────────┐           ┌──────────────────────┐
│  AWS Lambda        │           │  ECS Fargate         │
│  (Event-driven)    │           │  (Long-running)      │
└────────────────────┘           └──────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              Bedrock AgentCore Runtime                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │           Mortgage Assistant Agent                     │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │    Orchestration Layer                           │  │  │
│  │  │  - Request validation                            │  │  │
│  │  │  - Tool selection and execution                  │  │  │
│  │  │  - Session policy enforcement                    │  │  │
│  │  │  - Response generation                           │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
       │            │              │              │
       ▼            ▼              ▼              ▼
┌──────────┐ ┌───────────┐ ┌────────────┐ ┌──────────────┐
│ Bedrock  │ │  Lambda   │ │ MemoryDB   │ │ CloudWatch   │
│ Runtime  │ │  Tools    │ │ (Sessions) │ │   Logs       │
└──────────┘ └───────────┘ └────────────┘ └──────────────┘
     │
     ├─ Claude 3.5 Sonnet
     ├─ Amazon Titan
     └─ Custom Models
```

### Data Flow with Session Policies

1. **Request Ingress**: Client authenticates via Cognito or IAM, request enters through API Gateway
2. **Session Policy Application**: AgentCore validates request against dynamically scoped session policies
3. **Agent Invocation**: Bedrock runtime initializes agent with customer-specific context
4. **Tool Orchestration**: Agent calls Lambda functions or Step Functions for business logic
5. **Model Inference**: Claude 3.5 Sonnet processes context and generates structured response
6. **Memory Persistence**: Conversation state saved to MemoryDB with encryption at rest
7. **Response Delivery**: Structured response returned with audit trail logged to CloudWatch
8. **Security Posture Update**: Session policy updated based on agent actions for dynamic least privilege

### Bedrock AgentCore Unique Capabilities

**Dynamic Session Policies**: Unlike static IAM policies, AgentCore applies session-scoped permissions that change based on conversation state. For example, a mortgage agent might gain access to rate calculation APIs only after user authentication is verified.

**IAM-Driven Authorization**: AgentCore integrates directly with AWS IAM, enabling existing security controls to apply to agent actions without custom authorization logic.

## Deployment Options

### Option 1: Serverless Deployment with Lambda

**Best for**: Event-driven workloads, low-latency requirements, sporadic usage patterns

**Advantages**:
- Zero infrastructure management
- Automatic scaling from zero to thousands of requests
- Pay-per-invocation pricing
- Built-in fault tolerance

**Architecture**:

```python
# Lambda handler with Mangum adapter
from mangum import Mangum
from fastapi import FastAPI

app = FastAPI()

@app.post("/agent")
async def invoke_agent(request: dict):
    # Agent logic here
    pass

# Lambda handler
handler = Mangum(app)
```

**Deployment**:

```bash
# Package Lambda function
pip install -t package -r requirements.txt
cd package && zip -r ../function.zip . && cd ..
zip -g function.zip src/**/*.py

# Deploy to Lambda
aws lambda create-function \
    --function-name mortgage-agent \
    --runtime python3.11 \
    --role arn:aws:iam::$ACCOUNT_ID:role/AgentLambdaRole \
    --handler main.handler \
    --zip-file fileb://function.zip \
    --timeout 300 \
    --memory-size 1024 \
    --environment Variables="{BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0}"
```

### Option 2: Container Deployment with ECS Fargate

**Best for**: Long-running agents, WebSocket connections, multi-agent systems

**Advantages**:
- Full control over runtime environment
- Support for complex dependencies
- Persistent connections
- Custom networking configurations

**Deployment**:

```bash
# Build Docker image
docker build -t mortgage-agent:latest .

# Push to ECR
aws ecr create-repository --repository-name mortgage-agent
docker tag mortgage-agent:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/mortgage-agent:latest
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/mortgage-agent:latest

# Create ECS task definition and service (see infrastructure section)
```

### Option 3: Hybrid Deployment with AWS Outposts

**Best for**: Regulated industries requiring on-premises data processing

Enables running Bedrock workloads on-premises while maintaining cloud connectivity.

## Use Case: Mortgage Assistant Agent

### Overview

A mortgage rate advisory agent demonstrates AgentCore's session policy capabilities. The agent helps customers:

1. Calculate mortgage affordability based on income and debt
2. Compare current rates from multiple lenders
3. Generate personalized pre-qualification estimates
4. Schedule appointments with human loan officers

### Business Requirements

- **Privacy**: Customer financial data must never leave the user's session
- **Compliance**: All rate quotes must include required federal disclosures
- **Security**: Access to customer credit data requires authentication
- **Audit**: All financial calculations must be logged for regulatory review

### Technical Architecture

```
User Request → API Gateway → Lambda → AgentCore Runtime
                                           ↓
                              Session Policy: Allow calculate_payment
                                           ↓
                              Bedrock (Claude 3.5 Sonnet)
                                           ↓
                              Tool: calculate_affordability()
                                           ↓
                              Session Policy: Allow get_rates (if authenticated)
                                           ↓
                              Tool: fetch_current_rates()
                                           ↓
                              Response + Compliance Disclosures
```

### Sample Implementation

```python
import boto3
from typing import Dict, Any
import json

# Initialize Bedrock client
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

class MortgageAgent:
    """Mortgage advisory agent with dynamic session policies."""
    
    def __init__(self, model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"):
        self.model_id = model_id
        self.bedrock = bedrock_runtime
    
    def calculate_affordability(self, income: float, debts: float, down_payment: float) -> Dict[str, Any]:
        """Calculate maximum affordable home price.
        
        Uses standard 28/36 rule for debt-to-income ratios.
        
        Args:
            income: Annual gross income
            debts: Monthly debt payments
            down_payment: Available down payment amount
            
        Returns:
            Dictionary with affordability analysis
        """
        monthly_income = income / 12
        
        # 28% rule: housing payment should not exceed 28% of monthly income
        max_housing_payment = monthly_income * 0.28
        
        # 36% rule: total debt should not exceed 36% of monthly income
        max_total_debt = monthly_income * 0.36
        max_housing_with_debt = max_total_debt - debts
        
        # Use the more conservative limit
        max_monthly_payment = min(max_housing_payment, max_housing_with_debt)
        
        # Estimate home price (assuming 30-year mortgage at 7% interest)
        # P = M / [r(1+r)^n / ((1+r)^n - 1)]
        rate_monthly = 0.07 / 12
        n_payments = 360
        
        if rate_monthly > 0:
            loan_amount = max_monthly_payment / (rate_monthly * (1 + rate_monthly)**n_payments / ((1 + rate_monthly)**n_payments - 1))
        else:
            loan_amount = max_monthly_payment * n_payments
        
        max_home_price = loan_amount + down_payment
        
        return {
            "max_home_price": round(max_home_price, 2),
            "max_monthly_payment": round(max_monthly_payment, 2),
            "loan_amount": round(loan_amount, 2),
            "down_payment": down_payment,
            "dti_ratio_with_max": round((max_monthly_payment + debts) / monthly_income, 3)
        }
    
    def fetch_current_rates(self, credit_score: int, loan_type: str = "conventional") -> Dict[str, Any]:
        """Fetch current mortgage rates from lenders.
        
        NOTE: This requires authentication in production.
        
        Args:
            credit_score: Customer credit score
            loan_type: Type of mortgage (conventional, FHA, VA)
            
        Returns:
            Dictionary with rate information
        """
        # In production, this would call actual rate APIs
        # Rates vary by credit score
        base_rate = 7.0
        
        if credit_score >= 760:
            rate_adjustment = -0.5
        elif credit_score >= 700:
            rate_adjustment = -0.25
        elif credit_score >= 640:
            rate_adjustment = 0
        else:
            rate_adjustment = 0.5
        
        current_rate = base_rate + rate_adjustment
        
        return {
            "loan_type": loan_type,
            "interest_rate": current_rate,
            "apr": current_rate + 0.125,  # APR includes fees
            "credit_score_used": credit_score,
            "rate_lock_days": 60,
            "disclaimer": "Rates subject to change. Final rate determined at underwriting."
        }
    
    def generate_prequalification(self, affordability: Dict, rates: Dict) -> Dict[str, Any]:
        """Generate pre-qualification estimate.
        
        Args:
            affordability: Results from calculate_affordability
            rates: Results from fetch_current_rates
            
        Returns:
            Pre-qualification summary
        """
        return {
            "max_loan_amount": affordability["loan_amount"],
            "estimated_rate": rates["interest_rate"],
            "estimated_monthly_payment": affordability["max_monthly_payment"],
            "max_home_price": affordability["max_home_price"],
            "prequalification_valid_days": 90,
            "next_steps": "Submit full application for underwriting review"
        }
    
    def run(self, user_input: str, session_context: Dict[str, Any] = None) -> str:
        """Process user request through Bedrock agent.
        
        Args:
            user_input: User's question or request
            session_context: Session state and authentication context
            
        Returns:
            Agent's response
        """
        # Build tool configuration
        tools = [
            {
                "toolSpec": {
                    "name": "calculate_affordability",
                    "description": "Calculate maximum affordable home price based on income and debts",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "income": {"type": "number", "description": "Annual gross income"},
                                "debts": {"type": "number", "description": "Monthly debt payments"},
                                "down_payment": {"type": "number", "description": "Available down payment"}
                            },
                            "required": ["income", "debts", "down_payment"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "fetch_current_rates",
                    "description": "Fetch current mortgage rates. Requires authenticated session.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "credit_score": {"type": "integer", "description": "Customer credit score"},
                                "loan_type": {"type": "string", "enum": ["conventional", "FHA", "VA"]}
                            },
                            "required": ["credit_score"]
                        }
                    }
                }
            }
        ]
        
        # Prepare request
        messages = [{"role": "user", "content": user_input}]
        
        system_prompt = """You are a knowledgeable mortgage advisory assistant.

Your capabilities:
- Calculate home affordability based on income and debts
- Provide current mortgage rates (requires authentication)
- Generate pre-qualification estimates
- Explain mortgage terms and processes

Guidelines:
- Always include required federal disclosures with rate quotes
- Never guarantee approval - explain that underwriting is required
- Be transparent about factors affecting rates
- Recommend speaking with a licensed loan officer for formal approval

Important compliance notes:
- All rate quotes must include: "Rates subject to change and final approval"
- Must disclose: "This is not a commitment to lend"
- Must inform customers of factors affecting final rate
"""
        
        # Call Bedrock with tool configuration
        response = self.bedrock.converse(
            modelId=self.model_id,
            messages=messages,
            system=[{"text": system_prompt}],
            toolConfig={"tools": tools},
            inferenceConfig={
                "maxTokens": 4096,
                "temperature": 0.7
            }
        )
        
        # Process response and handle tool calls
        stop_reason = response['stopReason']
        
        if stop_reason == 'tool_use':
            # Extract tool calls from response
            tool_calls = [block for block in response['output']['message']['content'] 
                         if 'toolUse' in block]
            
            for tool_call in tool_calls:
                tool_name = tool_call['toolUse']['name']
                tool_input = tool_call['toolUse']['input']
                
                # Execute tool
                if tool_name == "calculate_affordability":
                    result = self.calculate_affordability(**tool_input)
                elif tool_name == "fetch_current_rates":
                    # Check authentication
                    if not session_context or not session_context.get('authenticated'):
                        result = {"error": "Authentication required to access rate information"}
                    else:
                        result = self.fetch_current_rates(**tool_input)
                
                # Continue conversation with tool result (simplified)
                # In production, this would continue the multi-turn conversation
                return json.dumps(result, indent=2)
        
        # Return text response
        text_content = [block['text'] for block in response['output']['message']['content'] 
                       if 'text' in block]
        return '\n'.join(text_content)
```

## Step-by-Step Implementation

### Step 1: Set Up Project Structure

```bash
# Create directory structure
mkdir -p mortgage-agent/{src,tests,infrastructure}
cd mortgage-agent

# Initialize git repository
git init

# Create .gitignore
cat > .gitignore << EOF
venv/
__pycache__/
*.pyc
.env
.aws-sam/
*.zip
EOF
```

### Step 2: Create Agent Configuration

Create `src/config.py`:

```python
"""Configuration management for mortgage agent."""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentConfig:
    """Agent configuration from environment variables."""
    
    aws_region: str
    bedrock_model_id: str
    max_tokens: int
    temperature: float
    session_timeout: int
    memorydb_endpoint: Optional[str]
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            session_timeout=int(os.getenv("SESSION_TIMEOUT", "3600")),
            memorydb_endpoint=os.getenv("MEMORYDB_ENDPOINT")
        )
```

### Step 3: Implement Tools

Create `src/tools/mortgage_tools.py`:

```python
"""Mortgage calculation and rate tools."""
from typing import Dict, Any
import logging
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getlogger(__name__)

class MortgageCalculator:
    """Mortgage calculation utilities."""
    
    @staticmethod
    def calculate_monthly_payment(
        principal: float,
        annual_rate: float,
        years: int
    ) -> float:
        """Calculate monthly mortgage payment.
        
        Args:
            principal: Loan amount
            annual_rate: Annual interest rate (as decimal, e.g., 0.07 for 7%)
            years: Loan term in years
            
        Returns:
            Monthly payment amount
        """
        monthly_rate = annual_rate / 12
        n_payments = years * 12
        
        if monthly_rate == 0:
            return principal / n_payments
        
        payment = principal * (
            monthly_rate * (1 + monthly_rate) ** n_payments
        ) / (
            (1 + monthly_rate) ** n_payments - 1
        )
        
        return round(payment, 2)
    
    @staticmethod
    def calculate_total_interest(
        principal: float,
        monthly_payment: float,
        years: int
    ) -> float:
        """Calculate total interest paid over loan term."""
        total_paid = monthly_payment * years * 12
        total_interest = total_paid - principal
        return round(total_interest, 2)
    
    @staticmethod
    def amortization_schedule(
        principal: float,
        annual_rate: float,
        years: int,
        num_periods: int = 12
    ) -> list[Dict[str, Any]]:
        """Generate amortization schedule.
        
        Args:
            principal: Loan amount
            annual_rate: Annual interest rate
            years: Loan term
            num_periods: Number of periods to return (default first year)
            
        Returns:
            List of payment periods with principal/interest breakdown
        """
        monthly_rate = annual_rate / 12
        monthly_payment = MortgageCalculator.calculate_monthly_payment(
            principal, annual_rate, years
        )
        
        balance = principal
        schedule = []
        
        for month in range(1, min(num_periods + 1, years * 12 + 1)):
            interest_payment = balance * monthly_rate
            principal_payment = monthly_payment - interest_payment
            balance -= principal_payment
            
            schedule.append({
                "month": month,
                "payment": round(monthly_payment, 2),
                "principal": round(principal_payment, 2),
                "interest": round(interest_payment, 2),
                "balance": round(balance, 2)
            })
        
        return schedule
```

### Step 4: Build Agent With Session Management

Create `src/agent.py`:

```python
"""Main mortgage agent implementation with session policies."""
import boto3
from typing import Dict, Any, List
import json
import logging
from src.config import AgentConfig
from src.tools.mortgage_tools import MortgageCalculator

logger = logging.getLogger(__name__)

class MortgageAssistantAgent:
    """Mortgage advisory agent with Bedrock AgentCore integration."""
    
    def __init__(self, config: AgentConfig):
        """Initialize agent with configuration.
        
        Args:
            config: Agent configuration object
        """
        self.config = config
        self.bedrock = boto3.client('bedrock-runtime', region_name=config.aws_region)
        self.calculator = MortgageCalculator()
        
    def _get_system_prompt(self) -> str:
        """Return agent system prompt with compliance guidelines."""
        return """You are an expert mortgage advisory assistant helping customers
understand home financing options.

Your responsibilities:
1. Calculate home affordability based on income and debts
2. Provide current mortgage rate information
3. Explain mortgage terms, requirements, and processes
4. Generate pre-qualification estimates
5. Help customers prepare for the application process

CRITICAL COMPLIANCE REQUIREMENTS:
- Always include disclosure: "Rates and terms subject to change. Not a commitment to lend."
- Remind customers that final approval requires underwriting review
- Explain that credit, income, and property appraisal affect final terms
- Recommend speaking with licensed loan officer for formal pre-approval

Be professional, accurate, and transparent about limitations."""
    
    def _build_tool_config(self, authenticated: bool = False) -> Dict[str, Any]:
        """Build tool configuration with session-based access control.
        
        Args:
            authenticated: Whether user is authenticated
            
        Returns:
            Tool configuration for Bedrock
        """
        tools = [
            {
                "toolSpec": {
                    "name": "calculate_affordability",
                    "description": "Calculate maximum affordable home price using income, debts, and down payment",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "annual_income": {
                                    "type": "number",
                                    "description": "Annual gross income in dollars"
                                },
                                "monthly_debts": {
                                    "type": "number",
                                    "description": "Total monthly debt payments in dollars"
                                },
                                "down_payment": {
                                    "type": "number",
                                    "description": "Available down payment in dollars"
                                }
                            },
                            "required": ["annual_income", "monthly_debts", "down_payment"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "calculate_monthly_payment",
                    "description": "Calculate monthly mortgage payment for a given loan amount and rate",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "loan_amount": {"type": "number"},
                                "annual_rate": {
                                    "type": "number",
                                    "description": "Annual interest rate as percentage (e.g., 7.0 for 7%)"
                                },
                                "loan_term_years": {"type": "integer", "description": "Loan term in years (typically 15 or 30)"}
                            },
                            "required": ["loan_amount", "annual_rate", "loan_term_years"]
                        }
                    }
                }
            }
        ]
        
        # Only include rate fetching tool if user is authenticated
        if authenticated:
            tools.append({
                "toolSpec": {
                    "name": "fetch_current_rates",
                    "description": "Fetch personalized current mortgage rates based on credit profile",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "credit_score": {"type": "integer", "description": "Credit score (300-850)"},
                                "loan_type": {
                                    "type": "string",
                                    "enum": ["conventional", "FHA", "VA", "USDA"],
                                    "description": "Type of mortgage loan"
                                },
                                "loan_amount": {"type": "number"}
                            },
                            "required": ["credit_score", "loan_type", "loan_amount"]
                        }
                    }
                }
            })
        
        return {"tools": tools}
    
    async def invoke(
        self,
        message: str,
        session_id: str,
        authenticated: bool = False
    ) -> Dict[str, Any]:
        """Invoke agent with user message.
        
        Args:
            message: User input
            session_id: Session identifier
            authenticated: Whether user is authenticated
            
        Returns:
            Agent response with tool results
        """
        logger.info(f"Processing message for session {session_id}")
        
        messages = [{"role": "user", "content": [{"text": message}]}]
        
        tool_config = self._build_tool_config(authenticated=authenticated)
        
        try:
            response = self.bedrock.converse(
                modelId=self.config.bedrock_model_id,
                messages=messages,
                system=[{"text": self._get_system_prompt()}],
                toolConfig=tool_config,
                inferenceConfig={
                    "maxTokens": self.config.max_tokens,
                    "temperature": self.config.temperature
                }
            )
            
            # Handle tool use
            if response['stopReason'] == 'tool_use':
                return await self._handle_tool_use(response, authenticated)
            
            # Extract text response
            content = response['output']['message']['content']
            text_response = ' '.join([
                block['text'] for block in content if 'text' in block
            ])
            
            return {
                "response": text_response,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error invoking agent: {e}")
            raise
    
    async def _handle_tool_use(
        self,
        response: Dict[str, Any],
        authenticated: bool
    ) -> Dict[str, Any]:
        """Handle tool execution from agent response."""
        # Implementation of tool execution logic
        # This would parse tool calls and execute corresponding methods
        pass
```

### Step 5: Create FastAPI Application

Create `src/main.py`:

```python
"""FastAPI application for mortgage agent."""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from mangum import Mangum
import logging
from src.agent import MortgageAssistantAgent
from src.config import AgentConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="Mortgage Assistant Agent API",
    description="AI-powered mortgage advisory service",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Initialize agent
config = AgentConfig.from_env()
agent = MortgageAssistantAgent(config)

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    session_id: str

class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Process chat message through agent.
    
    Args:
        request: Chat request with message and session ID
        credentials: Bearer token for authentication
        
    Returns:
        Agent response
    """
    try:
        # Verify authentication (simplified - use Cognito in production)
        authenticated = credentials.credentials is not None
        
        result = await agent.invoke(
            message=request.message,
            session_id=request.session_id,
            authenticated=authenticated
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mortgage-agent"}

# Lambda handler
handler = Mangum(app)
```

### Step 6: Create Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY main.py .

# Configure runtime
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Step 7: Deploy with AWS SAM

Create `template.yaml` for SAM deployment:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Mortgage Assistant Agent

Globals:
  Function:
    Timeout: 300
    MemorySize: 1024
    Runtime: python3.11
    Environment:
      Variables:
        BEDROCK_MODEL_ID: anthropic.claude-3-5-sonnet-20241022-v2:0
        AWS_REGION: us-east-1

Resources:
  MortgageAgentFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: .
      Handler: src.main.handler
      Policies:
        - AmazonBedrockFullAccess
        - CloudWatchLogsFullAccess
      Events:
        ChatApi:
          Type: Api
          Properties:
            Path: /chat
            Method: post

Outputs:
  ApiEndpoint:
    Description: "API Gateway endpoint URL"
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/chat/"
```

Deploy with SAM:

```bash
# Build
sam build

# Deploy
sam deploy --guided
```

## Advanced Features

### Session State Management with MemoryDB

```python
import redis
import json
from typing import Dict, Any

class SessionManager:
    """Manage agent sessions with MemoryDB."""
    
    def __init__(self, redis_endpoint: str, redis_port: int = 6379):
        self.redis = redis.Redis(
            host=redis_endpoint,
            port=redis_port,
            decode_responses=True
        )
    
    def save_session(self, session_id: str, data: Dict[str, Any], ttl: int = 3600):
        """Save session data with expiration."""
        self.redis.setex(
            f"session:{session_id}",
            ttl,
            json.dumps(data)
        )
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Retrieve session data."""
        data = self.redis.get(f"session:{session_id}")
        return json.loads(data) if data else {}
```

### Dynamic Session Policies

```python
def create_session_policy(user_authenticated: bool, credit_check_approved: bool) -> dict:
    """Generate session-scoped IAM policy for agent actions.
    
    This demonstrates AgentCore's unique capability to apply dynamic,
    conversation-state-dependent permissions.
    """
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream"
                ],
                "Resource": "*"
            }
        ]
    }
    
    # Grant rate access only if authenticated
    if user_authenticated:
        policy["Statement"].append({
            "Effect": "Allow",
            "Action": "lambda:InvokeFunction",
            "Resource": "arn:aws:lambda:*:*:function:fetch-mortgage-rates"
        })
    
    # Grant credit report access only if credit check authorized
    if credit_check_approved:
        policy["Statement"].append({
            "Effect": "Allow",
            "Action": "lambda:InvokeFunction",
            "Resource": "arn:aws:lambda:*:*:function:get-credit-report"
        })
    
    return policy
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel, Field
from typing import List

class AffordabilityReport(BaseModel):
    """Structured affordability analysis."""
    max_home_price: float = Field(description="Maximum affordable home price")
    max_monthly_payment: float
    recommended_down_payment: float
    debt_to_income_ratio: float
    assessment: str = Field(description="Qualitative assessment of affordability")
    recommendations: List[str]

# Use with agent
response_format = {
    "type": "json_schema",
    "json_schema": AffordabilityReport.model_json_schema()
}
```

## Security and Governance

### VPC Configuration for Private Deployment

```python
# CloudFormation for VPC setup
vpc_config = {
    "SecurityGroupIds": ["sg-xxxxx"],
    "SubnetIds": ["subnet-xxxxx", "subnet-yyyyy"]
}

# Lambda with VPC
lambda_function = {
    "FunctionName": "mortgage-agent",
    "VpcConfig": vpc_config,
    # ... other configuration
}
```

### Encryption at Rest and In Transit

```yaml
# SAM template with encryption
MortgageAgentFunction:
  Type: AWS::Serverless::Function
  Properties:
    Environment:
      Variables:
        BEDROCK_ENDPOINT: !Sub "https://bedrock-runtime.${AWS::Region}.amazonaws.com"
    KmsKeyArn: !GetAtt EncryptionKey.Arn

EncryptionKey:
  Type: AWS::KMS::Key
  Properties:
    Description: "Encryption key for agent data"
    KeyPolicy:
      Statement:
        - Effect: Allow
          Principal:
            Service: lambda.amazonaws.com
          Action:
            - kms:Decrypt
            - kms:Encrypt
          Resource: "*"
```

### Audit Logging

```python
import boto3
import json

cloudtrail = boto3.client('cloudtrail')

def log_agent_action(action: str, user_id: str, details: dict):
    """Log agent actions to CloudTrail for audit."""
    cloudtrail.put_insights_selectors(
        TrailName='agent-audit-trail',
        InsightSelectors=[
            {
                'InsightType': 'ApiCallRateInsight'
            }
        ]
    )
    
    # CloudWatch Logs for detailed agent interactions
    logs = boto3.client('logs')
    logs.put_log_events(
        logGroupName='/aws/agent/mortgage-assistant',
        logStreamName=user_id,
        logEvents=[{
            'timestamp': int(time.time() * 1000),
            'message': json.dumps({
                'action': action,
                'user_id': user_id,
                'details': details
            })
        }]
    )
```

## Monitoring and Observability

### CloudWatch Metrics

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

def publish_agent_metrics(metric_name: str, value: float, unit: str = 'Count'):
    """Publish custom metrics to CloudWatch."""
    cloudwatch.put_metric_data(
        Namespace='MortgageAgent',
        MetricData=[{
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Timestamp': datetime.utcnow()
        }]
    )

# Track agent performance
publish_agent_metrics('ResponseTime', response_time_ms, 'Milliseconds')
publish_agent_metrics('ToolInvocations', 3, 'Count')
publish_agent_metrics('TokensUsed', 1500, 'Count')
```

### X-Ray Tracing

```python
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.ext.flask.middleware import XRayMiddleware

# Instrument application
xray_recorder.configure(service='mortgage-agent')
XRayMiddleware(app, xray_recorder)

# Add custom segments
@xray_recorder.capture('bedrock_invocation')
def invoke_bedrock(message: str):
    # Your Bedrock call here
    pass
```

## Cost Optimization

### Model Selection Strategy

| Workload | Recommended Model | Cost Consideration |
|----------|-------------------|-------------------|
| Simple affordability calculations | Claude 3.5 Sonnet | Lowest cost, fast |
| Complex financial analysis | Claude 3.5 Sonnet (extended context) | Medium cost |
| Document analysis (tax returns) | Claude Opus | Higher cost, best accuracy |

### Request Optimization

```python
def optimize_prompt(user_message: str, conversation_history: list) -> str:
    """Optimize prompt to reduce token usage."""
    # Keep only last 10 messages
    recent_history = conversation_history[-10:]
    
    # Summarize older context
    if len(conversation_history) > 10:
        summary = "Previous conversation summary: User asked about mortgage rates and affordability."
        context = summary + "\n" + "\n".join(recent_history)
    else:
        context = "\n".join(recent_history)
    
    return f"{context}\n\nUser: {user_message}"
```

### Lambda Cold Start Mitigation

```python
# Use provisioned concurrency
import boto3

lambda_client = boto3.client('lambda')

lambda_client.put_provisioned_concurrency_config(
    FunctionName='mortgage-agent',
    ProvisionedConcurrentExecutions=2,  # Keep 2 instances warm
    Qualifier='$LATEST'
)
```

## Troubleshooting

### Common Issues

**Issue**: `AccessDeniedException` when calling Bedrock

**Solution**:
```bash
# Verify model access
aws bedrock list-foundation-models --region us-east-1

# Check IAM permissions
aws iam simulate-principal-policy \
    --policy-source-arn arn:aws:iam::$ACCOUNT_ID:role/AgentLambdaRole \
    --action-names bedrock:InvokeModel \
    --resource-arns "*"
```

**Issue**: Lambda timeout on long agent conversations

**Solution**:
```python
# Increase timeout and enable streaming
lambda_config = {
    "Timeout": 300,  # 5 minutes
    "MemorySize": 2048  # More memory = faster CPU
}

# Use streaming responses
response = bedrock.invoke_model_with_response_stream(
    modelId=model_id,
    body=request_body
)
```

**Issue**: High costs from excessive tool calls

**Solution**:
```python
# Implement tool call throttling
class ToolThrottler:
    def __init__(self, max_calls_per_session: int = 10):
        self.max_calls = max_calls_per_session
        self.call_counts = {}
    
    def can_call_tool(self, session_id: str) -> bool:
        count = self.call_counts.get(session_id, 0)
        if count >= self.max_calls:
            return False
        self.call_counts[session_id] = count + 1
        return True
```

### Debugging Tips

Enable debug logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log all Bedrock requests/responses
boto3.set_stream_logger('boto3.resources', logging.DEBUG)
```

## References

### Official Documentation

1. AWS. (2026). "Bedrock AgentCore Documentation." https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html

2. AWS. (2026). "Building Secure Multi-Agent Systems on AWS." https://aws.amazon.com/blogs/machine-learning/building-secure-multi-agent-systems

3. AWS. (2026). "Strands Agent SDK for Python." https://docs.strandshq.com/agent-sdk/python/quickstart

### Tutorials

4. AWS. (2026). "Deploy AI Agents with Lambda and Bedrock." https://aws.amazon.com/tutorials/deploy-ai-agents-lambda/

5. AWS. (2026). "Implementing Session Policies for AgentCore." https://aws.amazon.com/blogs/security/session-policies-agentcore

### Security Best Practices

6. AWS. (2026). "Security Patterns for Agentic Applications." https://docs.aws.amazon.com/prescriptive-guidance/latest/security-patterns-ai-agents/

7 AWS. (2026). "IAM Best Practices for AI Agents." https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices-ai-agents.html

### Sample Code

8. AWS Samples. (2026). "Bedrock AgentCore Examples." https://github.com/aws-samples/amazon-bedrock-agentcore-examples

9. Strands. (2026). "Agent SDK Sample Applications." https://github.com/StrandsHQ/agent-sdk-examples

---

*Last Updated: April 2026*
