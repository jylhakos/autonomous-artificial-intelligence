# How to build production-ready AI agents?

## Introduction

For AI agents to be useful, they need to interact with tools, data sources, and other AI agents. Two protocols offer standardized approaches to these connections. Anthropic's Model Context Protocol (MCP) gives AI agents a standardized way to connect with external data sources and stateless tools. Instead of building integrations for every service, developers can use MCP's standardized interface. Google's Agent2Agent Protocol (A2A) takes this further by enabling AI agents to communicate directly with each other, regardless of their underlying frameworks.

The Model Context Protocol is an open standard that enables developers to build secure, two-way connections between their data sources and AI-powered tools. MCP addresses the challenge of fragmented integrations by providing a universal, open standard for connecting AI systems with data sources, replacing fragmented integrations with a single protocol. The result is a simpler, more reliable way to give AI systems access to the data they need.

MCP-enabled systems can:
- **Standardize tool interfaces**: AI agents can discover and invoke tools at runtime without manual wiring through self-describing capabilities
- **Enable interoperability**: Tools become portable across different agent frameworks and hosting environments while maintaining governance
- **Simplify integration**: Developers build against a standard protocol instead of maintaining separate connectors for each data source

The Agent2Agent (A2A) Protocol extends these capabilities by enabling AI agents to communicate directly with each other through secure and structured message exchanges. A2A allows agents to discover each other's capabilities, negotiate how they will interact, and collaborate on tasks—regardless of their underlying frameworks or hosting environments.

### Key Resources

- [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
- [What is the Model Context Protocol (MCP)?](https://modelcontextprotocol.io/docs/getting-started/intro)
- [How to build production-ready AI agents with Google-managed MCP servers](https://cloud.google.com/blog/products/ai-machine-learning/how-to-build-ai-agents-with-google-managed-mcp-servers)
- [An AI agent example using Google MCP server with ADK](https://github.com/danistrebel/adk-cityscape)

## Deployment Architecture Overview

This documentation covers four primary deployment approaches for AI agents:

1. **Google Cloud Platform (GCP)**: Vertex AI Agent Engine, Cloud Run, and ADK-based deployments
2. **Amazon Web Services (AWS)**: Amazon Bedrock AgentCore with serverless architectures
3. **Microsoft Azure**: Azure AI Foundry with hosted agents and Agent Framework
4. **Local Deployment**: Docker containers with Ollama for private, cost-effective AI agents

Each approach offers distinct advantages depending on requirements for scalability, security, cost, and infrastructure control.

## 📁 Project Structure

```
deployment/
├── 📄 README.md                    # This file - overview and introduction
├── 📄 Google-GCP.md                # Google Cloud Platform deployment guide
├── 📄 Amazon-AWS.md                # Amazon Web Services deployment guide
├── 📄 Microsoft-Azure.md           # Microsoft Azure deployment guide
├── 📄 Docker-Ollama.md             # Local deployment with Docker and Ollama
├── 📁 sample-agents/               # Sample AI agent implementations
│   ├── 📁 gcp-customer-support/   # GCP agent example
│   ├── 📁 aws-mortgage-assistant/  # AWS agent example
│   ├── 📁 azure-research-agent/    # Azure agent example
│   └── 📁 local-ollama-agent/      # Local agent example
└── 📁 scripts/                     # Utility scripts for deployment
    ├── setup-venv.sh               # Virtual environment setup
    └── requirements.txt            # Python dependencies
```

## Google Cloud Platform (GCP)

Google Cloud offers tools for building production-ready AI agents through **Vertex AI Agent Engine**, **Cloud Run**, and the **Agent Development Kit (ADK)**. GCP's managed infrastructure handles scaling, security, and session memory automatically.

**Key Features:**
- Google-managed MCP servers with automatic context key injection
- Production-ready infrastructure with zero operations overhead
- Native integration with Google Cloud security stack (Cloud IAM, VPC-SC, Model Armor)
- Serverless deployment options via Cloud Run
- Integrated observability through Cloud Audit Logs

**Primary Deployment Methods:**
1. **Managed Option (Vertex AI Agent Engine)**: Deploy directly to Agent Engine with infrastructure handled by Google
2. **Serverless Option (Cloud Run)**:

 Container-based deployment as a web service

**Getting Started:**
- [Host AI agents on Cloud Run](https://docs.cloud.google.com/run/docs/ai-agents)
- [Deploy an agent](https://docs.cloud.google.com/agent-builder/agent-engine/deploy)
- [A developer's guide to production-ready AI agents](https://cloud.google.com/blog/products/ai-machine-learning/a-devs-guide-to-production-ready-ai-agents)

**Hands-on Labs:**
- [Local to Cloud: Full-Stack App Migration with Gemini CLI and Cloud SQL MCP](https://codelabs.developers.google.com/ai-mcp-dk-csql)
- [Getting Started with Agent2Agent (A2A) Protocol](https://codelabs.developers.google.com/intro-a2a-purchasing-concierge)
- [Build Your First ADK Agent Workforce](https://cloud.google.com/blog/topics/developers-practitioners/build-your-first-adk-agent-workforce)

**Detailed Setup Instructions**: See [Google-GCP.md](Google-GCP.md)

## Amazon Web Services (AWS)

AWS provides an ecosystem for deploying AI agents through **Amazon Bedrock AgentCore**, offering framework-agnostic environments with strong security and governance controls.

**Key Features:**
- Framework-agnostic execution environment supporting multiple agent frameworks
- Session isolation with dedicated microVMs for each user session
- Support for both real-time interactions and long-running workloads (up to 8 hours)
- Built-in capabilities for authentication and observability
- Integration with AWS security services (IAM, SCPs, permission boundaries)

**Primary Deployment Methods:**
1. **AgentCore Runtime**: Managed serverless platform for containerized agents
2. **AWS Lambda**: Serverless function-based agent deployment
3. **Amazon EC2/EKS**: Custom infrastructure deployment with full control

**Getting Started:**
- [Deploy AI agents on Amazon Bedrock AgentCore using GitHub Actions](https://aws.amazon.com/blogs/machine-learning/deploy-ai-agents-on-amazon-bedrock-agentcore-using-github-actions/)
- [Deploy and use an Amazon Bedrock agent in your application](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-deploy.html)
- [Secure AI agent access patterns to AWS resources using Model Context Protocol](https://aws.amazon.com/blogs/security/secure-ai-agent-access-patterns-to-aws-resources-using-model-context-protocol/)

**Architecture Resources:**
- [Effectively building AI agents on AWS Serverless](https://aws.amazon.com/blogs/compute/effectively-building-ai-agents-on-aws-serverless/)
- [Architecting for agentic AI development on AWS](https://aws.amazon.com/blogs/architecture/architecting-for-agentic-ai-development-on-aws/)

**Detailed Setup Instructions**: See [Amazon-AWS.md](Amazon-AWS.md)

## Microsoft Azure

Microsoft Azure offers production-grade AI agent development through **Azure AI Foundry** and the **Microsoft Agent Framework**, providing enterprise-ready tools with strong governance and observability.

**Key Features:**
- Hosted agent infrastructure with automatic scaling
- Enterprise-grade tool management with Model Context Protocol support
- Integration with Microsoft 365 ecosystem (Teams, SharePoint, Copilot)
- Built-in tools for common enterprise needs (web search, data analysis, browser automation)
- Native integration with Microsoft Entra ID for authentication and authorization

**Primary Deployment Methods:**
1. **Azure AI Foundry Hosted Agents**: Fully managed agent hosting with built-in monitoring
2. **Azure App Service**: Long-running agent deployment with Microsoft Agent Framework
3. **Azure Container Apps**: Scalable containerized agent deployment

**Getting Started:**
- [Develop AI agents on Azure](https://learn.microsoft.com/en-us/training/paths/develop-ai-agents-azure/)
- [Agent Factory: Building your first AI agent with the tools to deliver real-world outcomes](https://azure.microsoft.com/en-us/blog/agent-factory-building-your-first-ai-agent-with-the-tools-to-deliver-real-world-outcomes/)
- [Quickstart: Deploy your first hosted agent](https://learn.microsoft.com/en-us/azure/foundry/agents/quickstarts/quickstart-hosted-agent)

**Additional Resources:**
- [Hosted Containers and AI Agent Solutions](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/hosted-containers-and-ai-agent-solutions/4500627)
- [Part 2: Build Long-Running AI Agents on Azure App Service with Microsoft Agent Framework](https://azure.github.io/AppService/2025/10/31/app-service-agent-framework-part-2.html)
- [How to build and train AI agents](https://www.microsoft.com/en-us/microsoft-copilot/copilot-101/build-ai-agents)

**Detailed Setup Instructions**: See [Microsoft-Azure.md](Microsoft-Azure.md)

## Local Deployment with Docker and Ollama

For organizations requiring complete data privacy, cost-effectiveness, or offline capabilities, local deployment using **Docker** and **Ollama** provides a secure alternative to cloud-based solutions.

**Key Features:**
- Complete data privacy with no external API calls
- Zero token costs for model inference
- Support for various open-source models with tool calling capabilities
- Containerized deployment for consistency across environments
- Integration with popular frameworks (LangChain, CrewAI, Langflow)

**Primary Components:**
1. **Ollama**: Local LLM runtime supporting tool-enabled models
2. **Docker**: Containerization for consistent deployment
3. **MCP Toolkit**: Docker MCP Gateway for standardized tool integration

**Supported Frameworks:**
- [LangChain](https://docs.langchain.com/oss/python/langchain/quickstart)
- [CrewAI](https://docs.crewai.com/)
- [Langflow](https://www.langflow.org/)
- [Strands Agents](https://strandsagents.com/)

**Getting Started:**
- [Use containers for generative AI development](https://docs.docker.com/guides/genai-pdf-bot/develop/)
- [Ollama CLI Reference](https://docs.ollama.com/cli)
- [Ollama Quickstart](https://docs.ollama.com/quickstart)
- [Building AI Agents with Docker MCP Toolkit](https://www.docker.com/blog/docker-mcp-ai-agent-developer-setup/)

**Additional Resources:**
- [Unlocking Local AI: Using Ollama with Agents](https://www.langflow.org/blog/local-ai-using-ollama-with-agents)
- [Open WebUI](https://docs.openwebui.com/)
- [Agent Zero AI Framework](https://github.com/agent0ai/agent-zero)
- [LocalAI](https://github.com/mudler/LocalAGI)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [LM Studio](https://lmstudio.ai/)

**Detailed Setup Instructions**: See [Docker-Ollama.md](Docker-Ollama.md)

## Virtual Environment Setup

Before installing tools, libraries, or executing commands for any of the deployment options, it is recommended to create and activate a Python virtual environment. This ensures dependency isolation and prevents conflicts with system-wide packages.

### Setting Up a Virtual Environment

**For VS Code:**

1. Open the project folder in VS Code
2. Open the integrated terminal (Terminal → New Terminal)
3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the virtual environment:

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

5. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Virtual Environment for Each Platform

Each deployment platform has specific setup requirements:

- **Google GCP**: See detailed setup in [Google-GCP.md](Google-GCP.md#virtual-environment-setup)
- **Amazon AWS**: See detailed setup in [Amazon-AWS.md](Amazon-AWS.md#virtual-environment-setup)
- **Microsoft Azure**: See detailed setup in [Microsoft-Azure.md](Microsoft-Azure.md#virtual-environment-setup)
- **Docker/Ollama**: See detailed setup in [Docker-Ollama.md](Docker-Ollama.md#virtual-environment-setup)

### Common Dependencies

Create a `requirements.txt` file with the following base dependencies:

```txt
# Core dependencies
python-dotenv>=1.0.0
pydantic>=2.0.0
requests>=2.31.0

# Cloud SDK dependencies (install based on your platform)
# google-cloud-aiplatform>=1.38.0  # For GCP
# boto3>=1.28.0                     # For AWS
# azure-ai-foundry>=0.1.0           # For Azure

# Agent frameworks (install based on your choice)
# langchain>=0.1.0
# crewai>=0.1.0
# strands-agents>=0.1.0
```

### Deactivating the Virtual Environment

When finished working, deactivate the virtual environment:

```bash
deactivate
```

## Cross-Platform Framework: Strands Agents SDK

The [Strands Agents SDK](https://strandsagents.com/) provides a framework-agnostic approach to building AI agents that can be deployed on any cloud provider or locally. Built from production systems inside Amazon, Strands offers:

- **Universal Compatibility**: Deploy on AWS, GCP, Azure, or local infrastructure
- **Model Agnostic**: Works with any LLM provider (OpenAI, Anthropic, Bedrock, Vertex AI, etc.)
- **Production-Ready**: Built-in observability, memory management, and error handling
- **Steering Mechanisms**: Middleware-based control over agent behavior beyond prompting

Strands achieves higher task accuracy through its steering framework, which allows developers to define rules as code rather than relying solely on prompts.

## Agent Use Cases and Sample Implementations

### Google GCP: Customer Support Agent

A customer support agent that retrieves order statuses from databases, searches knowledge bases, and escalates complex issues to humans. This use case demonstrates multi-tool orchestration with Google's Gemini models.

**Key Capabilities:**
- Order status retrieval through database queries
- Knowledge base search for product information
- Automatic escalation routing based on complexity

**Deployment Options:**
- Vertex AI Agent Engine (managed)
- Cloud Run (serverless container)

See [Google-GCP.md](Google-GCP.md#use-case-customer-support-agent) for implementation details.

### Amazon AWS: Mortgage Assistant Agent

A mortgage application assistant that guides customers through the application process, answers questions from a knowledge base, and performs credit checks through secure integrations.

**Key Capabilities:**
- Multi-step workflow orchestration
- Secure integration with financial services APIs
- Knowledge base integration for policy information

**Deployment Options:**
- Amazon Bedrock AgentCore Runtime
- AWS Lambda with AgentCore SDK

See [Amazon-AWS.md](Amazon-AWS.md#use-case-mortgage-assistant-agent) for implementation details.

### Microsoft Azure: Automated Research Assistant

A research automation agent that uses web search and scraping tools to collect data, performs reasoning, and delivers structured responses following organizational guidelines.

**Key Capabilities:**
- Web data collection through integrated tools
- Multi-step reasoning and analysis
- Structured output formatting

**Deployment Options:**
- Azure AI Foundry Hosted Agents
- Azure App Service with Microsoft Agent Framework

See [Microsoft-Azure.md](Microsoft-Azure.md#use-case-automated-research-assistant) for implementation details.

### Local Deployment: Privacy-Focused Document Analysis

A document analysis agent running entirely on local infrastructure using Ollama and Docker, ensuring complete data privacy without external API dependencies.

**Key Capabilities:**
- PDF document parsing and analysis
- Question-answering over document content
- Zero external dependencies for data processing

**Deployment Options:**
- Docker Compose orchestration
- Standalone Ollama deployment

See [Docker-Ollama.md](Docker-Ollama.md#use-case-document-analysis-agent) for implementation details.

## Security and Governance Considerations

### Model Context Protocol Security

When deploying agents with MCP integrations, consider the following security principles:

1. **Principle 1: Assume all granted permissions could be used**
   - Design permissions based on acceptable scope of impact
   - Grant least-privilege access with resource-level restrictions
   - Implement monitoring for sensitive operations

2. **Principle 2: Provide organizational guidance on role usage**
   - Use session policies to scope permissions per tool invocation
   - Implement permission boundaries at organizational level
   - Tag agent roles consistently for governance and audit

3. **Principle 3: Differentiate AI-driven from human-initiated actions**
   - Use AWS-managed MCP context keys or self-managed session tags
   - Apply different IAM rules based on actor type
   - Maintain audit trails for compliance investigations

Source: [Secure AI agent access patterns to AWS resources using Model Context Protocol](https://aws.amazon.com/blogs/security/secure-ai-agent-access-patterns-to-aws-resources-using-model-context-protocol/)

### Data Privacy and Compliance

Each deployment option offers different levels of data control:

- **Cloud Deployments (GCP, AWS, Azure)**: Data processed within cloud provider infrastructure with enterprise security controls
- **Local Deployments (Docker/Ollama)**: Complete data isolation with no external transmission

## Learning Resources and Community

### Online Courses and Training

- [AI Agents for Beginners - Microsoft Course](https://github.com/microsoft/ai-agents-for-beginners)
- [Production-Ready AI with Google Cloud Learning Path](https://cloud.google.com/blog/topics/developers-practitioners/production-ready-ai-with-google-cloud-learning-path)
- [Amazon Bedrock Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/a4bdb007-5600-4368-81c5-ff5b4154f518/en-US)
- [Introduction to AI in Azure](https://learn.microsoft.com/en-us/training/paths/introduction-to-ai-on-azure/)

### Observability and Monitoring

- [LangSmith: Know what your agents are really doing](https://www.langchain.com/langsmith/observability)
- [Azure AI Foundry Tracing](https://learn.microsoft.com/en-us/azure/ai-foundry/agents/concepts/tracing)
- [Google Cloud Audit Logs for MCP](https://cloud.google.com/blog/products/ai-machine-learning/how-to-build-ai-agents-with-google-managed-mcp-servers)

### Security Best Practices

- [From Hallucinations to Prompt Injection: Securing AI Workflows at Runtime](https://www.docker.com/blog/secure-ai-agents-runtime-security/)
- [Creating a blueprint for safe and secure AI agents](https://azure.microsoft.com/en-us/blog/agent-factory-creating-a-blueprint-for-safe-and-secure-ai-agents/)
- [AWS Agentic AI Scoping Matrix](https://aws.amazon.com/ai/security/agentic-ai-scoping-matrix/)

## Advanced Topics

### Multi-Agent Systems

Multi-agent systems enable collaboration between specialized agents to handle complex tasks:

- [Build Multi-Agent Systems with ADK](https://codelabs.developers.google.com/codelabs/production-ready-ai-with-gc/3-developing-agents/build-a-multi-agent-system-with-adk)
- [Create Expert Content: Deploying a Multi-Agent System with Terraform and Cloud Run](https://cloud.google.com/blog/topics/developers-practitioners/create-expert-content-deploying-a-multi-agent-system-with-terraform-and-cloud-run)
- [Orchestrate a multi-agent solution using the Microsoft Agent Framework](https://learn.microsoft.com/en-us/training/modules/orchestrate-semantic-kernel-multi-agent-solution/)

### Agent-to-Agent Communication

The Agent2Agent (A2A) Protocol enables agents to discover, negotiate, and collaborate:

- [Discover Azure AI Agents with A2A](https://learn.microsoft.com/en-us/training/modules/discover-agents-with-a2a/)
- [Getting Started with A2A Protocol: Purchasing Concierge Example](https://codelabs.developers.google.com/intro-a2a-purchasing-concierge)

### Production Deployment Patterns

- [From Prototype to Production - Azure AI Foundry](https://azure.microsoft.com/en-us/blog/agent-factory-from-prototype-to-production-developer-tools-and-rapid-agent-development/)
- [Deploy an agent - Google Cloud](https://docs.cloud.google.com/agent-builder/agent-engine/deploy)
- [How to Build, Run, and Package AI Models Locally with Docker Model Runner](https://www.docker.com/blog/how-to-build-run-and-package-ai-models-locally-with-docker-model-runner/)

## Conclusion

This documentation provides tutorial for deploying AI agents across multiple platforms—from fully managed cloud services to private local deployments. Each platform offers distinct advantages:

- **Google GCP** excels in managed infrastructure and integration with Google services
- **Amazon AWS** provides security controls and enterprise governance
- **Microsoft Azure** offers deep integration with Microsoft 365 and enterprise tooling
- **Docker/Ollama** enables complete privacy and cost control for local deployments

The choice of platform depends on specific requirements for scalability, security, compliance, cost, and operational control. By leveraging standardized protocols like MCP and A2A, organizations can build portable, interoperable agent systems that work across these environments.

For detailed step-by-step instructions specific to each platform, refer to the deployment guides:

- [Google-GCP.md](Google-GCP.md)
- [Amazon-AWS.md](Amazon-AWS.md)
- [Microsoft-Azure.md](Microsoft-Azure.md)
- [Docker-Ollama.md](Docker-Ollama.md)

**Install platform-specific dependencies**:
   ```bash
   # For GCP agent
   pip install -r sample-agents/gcp-customer-support/requirements.txt
   
   # For AWS agent
   pip install -r sample-agents/aws-mortgage-assistant/requirements.txt
   
   # For Azure agent
   pip install -r sample-agents/azure-research-agent/requirements.txt
   
   # For local Ollama agent
   pip install -r sample-agents/local-ollama-agent/requirements.txt
   ```

## References

### Primary Documentation

1. Anthropic. (2024). "Introducing the Model Context Protocol." Retrieved from https://www.anthropic.com/news/model-context-protocol

2. Model Context Protocol. (2024). "What is the Model Context Protocol (MCP)?" Retrieved from https://modelcontextprotocol.io/docs/getting-started/intro

3. Google Cloud. (2026). "How to build production-ready AI agents with Google-managed MCP servers." Retrieved from https://cloud.google.com/blog/products/ai-machine-learning/how-to-build-ai-agents-with-google-managed-mcp-servers

4. Amazon Web Services. (2026). "Deploy AI agents on Amazon Bedrock AgentCore using GitHub Actions." Retrieved from https://aws.amazon.com/blogs/machine-learning/deploy-ai-agents-on-amazon-bedrock-agentcore-using-github-actions/

5. Microsoft Azure. (2025). "Agent Factory: Building your first AI agent with the tools to deliver real-world outcomes." Retrieved from https://azure.microsoft.com/en-us/blog/agent-factory-building-your-first-ai-agent-with-the-tools-to-deliver-real-world-outcomes/

6. Docker, Inc. (2025). "Building AI Agents with Docker MCP Toolkit: A Developer's Real-World Setup." Retrieved from https://www.docker.com/blog/docker-mcp-ai-agent-developer-setup/

### Academic and Research Resources

7. Strebel, D., & Shen, L. (2026). "ADK Cityscape: An AI agent example using Google MCP server with ADK." GitHub repository. Retrieved from https://github.com/danistrebel/adk-cityscape

8. LangChain. (2026). "Build AI agents you can actually steer: Strands Agents SDK." Retrieved from https://strandsagents.com/

9. Ollama. (2024). "Ollama Documentation." Retrieved from https://docs.ollama.com/

10. Microsoft. (2025). "AI Agents for Beginners - A Course." GitHub repository. Retrieved from https://github.com/microsoft/ai-agents-for-beginners

---

**Note**: All referenced technologies and platforms are subject to change. Always consult official documentation for the most current information.

---

*Last Updated: April 2026*
