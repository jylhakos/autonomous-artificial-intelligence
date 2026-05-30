# Autonomous Artificial Intelligence

A tutorial for building and managing autonomous AI agents with advanced capabilities in collaboration, orchestration, evaluation, and observability. This repository explores collaborative AI agents and multi-agent systems and provides practical examples, step-by-step tutorials, and testing methodologies for building intelligent, autonomous agents that work together to solve tasks.

## Table of Contents

- [Folders Structure](#folders-structure)
  - [agents/](#-agents)
  - [autonomous/](#-autonomous)
  - [collaboration/](#-collaboration)
  - [datasets/](#-datasets)
  - [deployment/](#-deployment)
  - [evaluation/](#-evaluation)
  - [fine-tuning/](#-fine-tuning)
  - [memory/](#-memory)
  - [models/](#-models)
  - [observability/](#-observability)
  - [optimization/](#-optimization)
  - [orchestration/](#-orchestration)
  - [platform/](#-platform)
  - [reasoning/](#-reasoning)
  - [retrieval/](#-retrieval)
  - [security/](#-security)
  - [training/](#-training)
  - [workflows/](#-workflows)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [License](#license)

---

## Folders Structure

```
📁 autonomous-artificial-intelligence/
├── 📂 agents/              # Single-agent, sub-agents, and multi-agents with Google ADK
├── 📂 autonomous/          # Core autonomous AI agent framework
├── 📂 collaboration/       # Multi-agent collaboration patterns
├── 📂 datasets/            # Dataset preparation, cleaning, and synthetic generation for LLMs and ML
├── 📂 deployment/          # Production deployment strategies and guides
├── 📂 evaluation/          # Agent performance evaluation system
├── 📂 fine-tuning/         # LLM fine-tuning with Unsloth, PEFT, LoRA, and QLoRA
├── 📂 memory/              # Context windows, short-term and long-term agent memory
├── 📂 models/              # Machine learning models, LLMs, and local inference server
├── 📂 observability/       # Monitoring and analysis tools
├── 📂 optimization/        # ML and LLM optimization techniques and tools
├── 📂 orchestration/       # Workflow orchestration and coordination
├── 📂 platform/            # MLflow platform for agents, LLMs, and ML models
├── 📂 reasoning/           # LLM reasoning techniques and training workflows
├── 📂 retrieval/           # Vector databases, vector search and RAG for LLMs
├── 📂 security/            # AI agent security and threat protection
├── 📂 training/            # Pre-training, post-training, alignment, and fine-tuning for LLMs and ML
└── 📂 workflows/           # SDLC integration with Vibe Coding and Foundry Local
```

### 📂 agents/

**Key Points:**
- Tutorial of three agent architectures: single-agent, sub-agents, and multi-agents
- Built entirely with **Google's Agent Development Kit (ADK)** — the open-source Python framework for building, testing, and deploying agents locally and on Google Cloud
- **Single-agent** pattern using `LlmAgent` for bounded, self-contained tasks (researcher example with Gemini 2.0 Flash)
- **Sub-agents** pattern using `SequentialAgent` (Researcher → Writer → Editor pipeline) and `ParallelAgent` (FlightFinder + HotelFinder fan-out)
- **Multi-agents** supervisor pattern with four specialist agents (`ResearchAgent`, `AnalysisAgent`, `WriterAgent`, `ReviewerAgent`) coordinated by an orchestrator
- Local development with ADK Web UI (`adk web`) and optional Ollama + Open WebUI for open-source LLM inference
- Decision framework for choosing the right architecture, referencing Google ADK and Microsoft Cloud Adoption Framework guidance

**Importance for Autonomous AI:**
The `agents/` folder provides the clearest conceptual entry point into autonomous AI architectures. It demonstrates how a single `LlmAgent` is the right default for bounded tasks, when sub-agents become necessary to delegate stateful sub-processes within a shared session, and how a full multi-agent system with isolated specialists and a supervisor enables independent scaling, compliance boundaries, and parallel development by multiple teams. By implementing all three patterns with the same Google ADK framework, the trade-offs in latency, cost, context management, and operational complexity are directly comparable at the code level — making this the practical companion to the architectural theory covered in the `autonomous/` and `collaboration/` folders.

**Key Components:**
- `single_agent/agent.py` — Single `LlmAgent` researcher example; the foundational pattern using `Runner` and `InMemorySessionService`
- `sub_agents/agent.py` — Sequential pipeline (Researcher → Writer → Editor) and parallel fan-out (FlightFinder + HotelFinder) sub-agent examples
- `multi_agents/agent.py` — Supervisor pattern with four domain-specialist agents coordinated by a `SequentialAgent` orchestrator
- `.env.example` — API key template (`GOOGLE_API_KEY`) for Gemini model access via Google AI Studio or Vertex AI
- `requirements.txt` — `google-adk`, `python-dotenv`, and runtime dependencies
- `README.md` — Architecture guide, orchestrator-worker decision framework, local setup (VS Code / Linux), ADK Web UI, Ollama integration, and prompt engineering tips

```
agents/
├── 📄 .env.example              API key template (GOOGLE_API_KEY)
├── 📄 requirements.txt          google-adk, python-dotenv, and dependencies
├── 📄 README.md                 Architecture guide and local setup
├── 📂 single_agent/
│   └── agent.py                 LlmAgent researcher — single bounded task
├── 📂 sub_agents/
│   └── agent.py                 SequentialAgent pipeline + ParallelAgent fan-out
└── 📂 multi_agents/
    └── agent.py                 Supervisor pattern with 4 specialist agents
```

---

### 📂 autonomous/

**Key Points:**
- Core framework for building autonomous AI agents
- Modular architecture with agents, memory, tools, and workflows
- Includes setup utilities and documentation
- Ready-to-run examples demonstrating various agent patterns

**Importance for Autonomous AI:**
This is the foundation of the entire system. It provides the essential building blocks for creating agents that can operate independently, maintain context through memory systems, utilize external tools, and execute complex workflows. The modular design allows agents to be composed and extended based on specific requirements.

**Key Components:**
- `agents/` - Agent implementations and base classes
- `memory/` - Context and state persistence for long-running agents
- `tools/` - External capabilities agents can leverage
- `workflows/` - Predefined execution patterns
- `examples/` - Practical demonstrations including multi-model chat, hybrid collaboration

### 📂 collaboration/

**Key Points:**
- Multi-agent collaboration frameworks (AutoGen, Claude)
- Patterns for agents working together on complex tasks
- Human-in-the-loop feedback mechanisms
- Test suite for collaboration scenarios

**Importance for Autonomous AI:**
Autonomous agents rarely operate in isolation. This folder provides patterns and implementations for agents to collaborate, negotiate, delegate tasks, and coordinate actions. It demonstrates how multiple specialized agents can work together to solve problems beyond the capability of individual agents, mirroring human team dynamics.

**Key Features:**
- AutoGen-based multi-agent systems
- Hybrid agent collaboration (combining different AI frameworks)
- Code collaboration and review workflows
- Human feedback integration for guided autonomy

### 📂 datasets/

**Key Points:**
- Reference and hands-on workspace for preparing, cleaning, and generating datasets for LLM and ML model training and fine-tuning
- Implements the **Self-Instruct** technique for synthetic instruction-tuning dataset creation using a local Ollama model
- Covers three distinct data preparation workflows: **pre-training corpora** (extraction, normalisation, deduplication, quality filtering), **instruction fine-tuning** (structured instruction-input-output triplets), and **preference fine-tuning** (RLHF/DPO chosen/rejected response pairs)
- Dataset formats covered: instruction-tuning triplets (`system`, `instruction`, `input`, `output`) and preference-tuning pairs (`prompt`, `chosen`, `rejected`) compatible with SFT and DPO training pipelines
- Dataset quality evaluation: near-duplicate detection, LLM-based quality scoring, statistical validation with TFDV, label error detection with Cleanlab, LLM-as-a-Judge, and discriminative testing
- Synthetic data generation pipeline: prompt generation, filtering, query evolution, and styling customization using LLMs as data factories

**Importance for Autonomous AI:**
Dataset quality is the deterministic factor in the behavior of autonomous AI agents. A model's ability to follow instructions, reason through multi-step problems, and refuse harmful requests is entirely encoded in the data it was trained on — not in architecture choices or compute budget alone. The Self-Instruct technique allows autonomous systems to bootstrap their own training data, creating a feedback loop where a base model generates candidate instructions, filters low-quality examples, and produces instruction-response pairs that teach instruction-following behavior at scale. For fine-tuning autonomous agents on domain-specific tasks — whether code generation, structured data extraction, or tool-use protocols — the data preparation workflows covered here (cleaning, deduplication, format conversion, quality scoring) determine whether the resulting model reliably executes the target behavior or reverts to generic responses. Preference datasets (RLHF/DPO) further shape how autonomous agents balance helpfulness, safety, and alignment with human values across edge cases that rule-based guardrails cannot anticipate.

**Key Components:**
- `scripts/seed_tasks.py` — Human-written seed task definitions that initialize the Self-Instruct generation pipeline
- `scripts/self_instruct_generator.py` — Full Self-Instruct pipeline using a local Ollama model to generate, filter, and format instruction-tuning datasets
- `scripts/dataset_cleaner.py` — Text cleaning, deduplication, and quality filtering for raw corpora and generated datasets
- `requirements.txt` — Python dependencies for dataset processing and generation workflows
- `README.md` — Document to describe data preparation for pre-training and fine-tuning, dataset formats, synthetic data generation, quality evaluation methods, and environment setup

```
datasets/
├── 📄 requirements.txt              Python dependencies
├── 📄 README.md                     Dataset preparation, synthetic generation, and quality guide
└── 📂 scripts/
    ├── seed_tasks.py                Human-written seed task definitions
    ├── self_instruct_generator.py   Full Self-Instruct pipeline with Ollama
    └── dataset_cleaner.py           Text cleaning, deduplication, quality filtering
```

---

### 📂 deployment/

**Key Points:**
- Production-ready deployment strategies for AI agents
- Model Context Protocol (MCP) for standardized tool and data source integration
- Agent2Agent (A2A) Protocol for inter-agent communication
- Cloud deployment options (GCP, AWS, Azure) with managed infrastructure
- Local deployment with Docker and Ollama for privacy and cost efficiency
- Sample agent implementations for each deployment platform

**Importance for Autonomous AI:**
Deploying autonomous AI agents to production requires careful consideration of infrastructure, security, scalability, and cost. This folder demonstrates multiple deployment approaches, each with distinct trade-offs. Cloud deployments (GCP Vertex AI, AWS Bedrock AgentCore, Azure AI Foundry) offer managed infrastructure with automatic scaling, built-in security, and zero operations overhead, making them ideal for enterprise applications requiring high availability and integration with cloud ecosystems. Local deployments using Docker and Ollama provide complete data privacy, zero inference costs, and offline capabilities, suitable for organizations with strict data residency requirements or cost constraints. The Model Context Protocol (MCP) standardizes how agents connect to tools and data sources across all platforms, while the Agent2Agent (A2A) Protocol enables cross-framework agent collaboration. Understanding these deployment patterns is essential for building autonomous AI systems that are reliable, secure, and production-ready.

**Key Features:**
- Tutorials for Google Cloud Platform (Vertex AI Agent Engine, Cloud Run, ADK)
- AWS deployment patterns with Amazon Bedrock AgentCore and serverless architectures
- Azure AI Foundry integration with Microsoft 365 ecosystem and enterprise tools
- Docker-based local deployment with open-source models via Ollama
- Sample agents demonstrating customer support, mortgage assistance, research, and local processing
- Deployment scripts and utilities for an environment setup
- Security best practices for cloud and local deployments

### 📂 evaluation/

**Key Points:**
- Systematic agent evaluation and benchmarking
- Performance metrics and quality assessment
- Online and offline evaluation capabilities
- Integration with observability tools for tracing

**Importance for Autonomous AI:**
Autonomous agents must be reliable and measurable. This folder provides the infrastructure to evaluate agent performance, track improvements, identify failure modes, and ensure quality. Without proper evaluation, autonomous systems cannot be trusted in production environments.

**Key Components:**
- `agent_evaluation.py` - Core evaluation framework
- `agent_with_tracing.py` - Observable agent implementations
- `online_evaluation.py` - Real-time performance monitoring
- Workflow guides for structured evaluation processes

### 📂 fine-tuning/

**Key Points:**
- Task-specific adaptation of pre-trained LLMs without training from scratch
- Parameter-Efficient Fine-Tuning (PEFT) with LoRA and QLoRA for low-memory training on consumer GPUs
- Unsloth integration providing 2x faster training with 60-80% less VRAM compared to standard Hugging Face training
- Supervised Fine-Tuning (SFT) on instruction-following datasets to encode persistent behavior into model weights
- Support for local environments (Linux/GPU, Docker with CUDA) and cloud deployment on Microsoft Azure
- Coverage of full fine-tuning versus PEFT trade-offs, gradient checkpointing, and Flash Attention

**Importance for Autonomous AI:**
Fine-tuning enables autonomous agents to develop specialized behavior encoded directly in model weights rather than relying on long prompts or in-context examples. An agent fine-tuned on domain-specific data consistently applies correct reasoning patterns, output formats, and tool-use conventions across every inference call, without consuming context window space. For autonomous systems, this translates to lower latency, reduced cost, and more reliable behavior in specialized domains such as code generation, structured data extraction, or instruction-following in production workflows. PEFT techniques like LoRA make this practical on consumer hardware by training only a small fraction of added parameters — typically less than 1% of total model weights — rather than updating all model weights, making fine-tuning accessible without large-scale GPU clusters. Unsloth further reduces the barrier by applying hand-written CUDA kernels and memory-efficient backpropagation to cut training time and VRAM requirements significantly.

**Key Components:**
- `scripts/unsloth_finetune.py` - Fast fine-tuning pipeline using Unsloth with LoRA and QLoRA adapters
- `scripts/peft_finetune.py` - Parameter-efficient fine-tuning using Hugging Face PEFT and LoRA
- `scripts/requirements.txt` - Dependency list for Unsloth, PEFT, Transformers, and related training libraries
- Unsloth - Optimized training framework with hand-written CUDA kernels delivering 2x faster throughput and 60-80% VRAM reduction
- PEFT (Parameter-Efficient Fine-Tuning) - Hugging Face library for training lightweight adapter layers while keeping base model weights frozen
- LoRA (Low-Rank Adaptation) - Injects trainable low-rank matrices into transformer attention layers, enabling fine-tuning on GPUs with 8-24 GB VRAM
- QLoRA - Combines 4-bit quantization of the base model with LoRA adapters, allowing large models (13B-70B) to be fine-tuned on a single consumer GPU
- Axolotl and LLaMA-Factory - Alternative fine-tuning frameworks covered in the documentation for production and multi-dataset workflows
- DeepSpeed - Distributed training integration for scaling fine-tuning across multiple GPUs

### 📂 memory/

**Key Points:**
- Practical guide and runnable code samples for understanding how AI agents use **context**, **short-term memory**, and **long-term persistent memory**
- Three-layer production-ready memory architecture: **context window** (active working memory), **retrieval layer** (RAG for semantic search), and **persistent memory store** (cross-session continuity)
- Short-term memory (in-context / session state): conversation turns, tool call results, and state variables — scoped to the active session
- Long-term persistent memory divided by nature: **episodic** (specific past events), **semantic** (general facts and user preferences), and **procedural** (workflows and behavioral rules)
- **Google ADK** samples demonstrating `InMemorySessionService` for short-term session state and `MemoryService` for persistent cross-session memory retrieval
- **Microsoft Foundry** samples demonstrating Foundry Memory Store for long-term persistence and Foundry Conversations API for short-term context management
- Context engineering concepts: context window limits, attention budget management, context compaction, and prefix caching strategies

**Importance for Autonomous AI:**
Without persistent memory, every conversation starts from a blank slate — agents cannot reference prior decisions, user preferences, or accumulated knowledge. Memory is what transforms a stateless LLM call into a context-aware autonomous agent capable of sustained, goal-directed behavior across multiple sessions. Short-term memory (the context window) determines what the agent can reason about in the current request; long-term memory determines whether the agent can learn from and build upon past interactions. For production autonomous systems, the three-layer memory architecture — context window, RAG retrieval, and persistent store — allows agents to operate at scale without context overflows, while retaining the continuity of behavior that makes them genuinely useful over time. Google ADK's `MemoryService` and Microsoft Foundry's Memory Store provide the concrete infrastructure to implement this at both local-development and cloud-production scales, covering the full spectrum from in-session state variables to cross-session user profiles and episodic event logs.

**Key Components:**
- `samples/adk/short_term_memory.py` — ADK session state demo using `InMemorySessionService` for in-session context and state variables
- `samples/adk/long_term_memory.py` — ADK `MemoryService` demo for persistent cross-session memory storage and retrieval
- `samples/adk/README.md` — Google ADK memory architecture guide covering session state, `MemoryService`, and context compaction
- `samples/foundry/foundry_memory.py` — Microsoft Foundry Memory Store demo for long-term persistent agent memory across sessions
- `samples/foundry/foundry_short_term_context.py` — Foundry Conversations API demo for managing short-term context and message threads
- `samples/foundry/README.md` — Foundry memory integration guide covering thread management, Memory Store, and deployment patterns
- `requirements.txt` — Python dependencies for Google ADK, Microsoft Foundry, and related memory integrations
- `README.md` — A guide covering context windows, all memory types, ADK and Foundry implementations, framework comparisons, and use cases

```
memory/
├── 📄 .env.example                        Environment variable template
├── 📄 requirements.txt                    Python dependencies
├── 📄 README.md                           Context and memory guide
└── 📂 samples/
    ├── 📂 adk/
    │   ├── short_term_memory.py           ADK session state demo (InMemorySessionService)
    │   ├── long_term_memory.py            ADK MemoryService demo (persistent cross-session)
    │   └── README.md                      ADK memory architecture and setup guide
    └── 📂 foundry/
        ├── foundry_memory.py              Foundry Memory Store demo (long-term persistence)
        ├── foundry_short_term_context.py  Foundry Conversations API (short-term context)
        └── README.md                      Foundry memory integration guide
```

---

### 📂 models/

**Key Points:**
- Covers two primary model types in modern AI systems: **Machine Learning (ML) models** (RNN, LSTM) for sequential and time-series tasks, and **Large Language Models (LLMs)** for language understanding and generation
- End-to-end pipeline for building a GPT-style LLM from scratch: tokenization, transformer architecture with multi-head attention, pre-training on a text corpus, supervised fine-tuning, and chatbot deployment
- Custom **FastAPI inference server** for serving the locally-built LLM via REST API endpoints, enabling fully private and offline-capable model deployment without third-party API dependencies
- **Nginx reverse proxy** with token-based authorization (`auth_request`) to secure the inference server and control access
- **Open WebUI** chat frontend integration providing a conversational interface to the locally-running model
- **RNN and LSTM** models for time-series forecasting of service request volumes, illustrating how classical ML complements LLMs in autonomous systems
- Service load forecasting pipeline: data preparation, LSTM training, next-day usage prediction, and visualization with evaluation metrics (MAE, RMSE)
- Request logging and CSV dataset generation from live inference traffic to feed back into ML training workflows

**Importance for Autonomous AI:**
Understanding how large language models are built from the ground up — tokenization, attention mechanisms, transformer blocks, pre-training objectives, and supervised fine-tuning — gives autonomous AI practitioners the architectural intuition required to diagnose failure modes, select the right model for a task, and make informed decisions about fine-tuning versus prompting versus RAG. Building and deploying a model's own inference server removes the dependency on third-party APIs, enabling fully private, offline-capable autonomous agents with deterministic latency and no per-token billing. The LSTM forecasting pipeline demonstrates how classical ML models complement LLMs within autonomous systems: while LLMs handle language understanding and generation, sequential models like LSTMs excel at predicting resource demand from historical time-series data, enabling autonomous agents to proactively adapt behavior or scale infrastructure based on anticipated load patterns. Together, the from-scratch LLM and the forecasting pipeline establish the full spectrum of model engineering knowledge that underpins every component in this repository.

**Key Components:**
- `scripts/llm_from_scratch/model.py` — GPT-style transformer model implementation with multi-head self-attention and feed-forward layers
- `scripts/llm_from_scratch/tokenizer.py` — Tokenizer for text preprocessing used during pre-training and inference
- `scripts/llm_from_scratch/train.py` — Pre-training script for the GPT-style model on a text corpus
- `scripts/llm_from_scratch/text_classifier.py` — Fine-tuning the pre-trained model as a text classifier
- `scripts/llm_from_scratch/chatbot.py` — Interactive chatbot using the pre-trained model for text generation
- `scripts/llm_from_scratch/api_server.py` — FastAPI inference server exposing the model via REST API endpoints with chat completion support
- `scripts/nginx/nginx.conf` — Nginx reverse proxy configuration with `auth_request`-based token authorization for securing the inference server
- `scripts/logging/log_analyzer.py` — Request logging and analysis tool that produces CSV datasets from live inference traffic for ML training
- `scripts/forecasting/prepare_data.py` — Data preparation and normalization pipeline for LSTM training
- `scripts/forecasting/train_lstm.py` — LSTM model training for service load time-series forecasting
- `scripts/forecasting/forecast.py` — Next-day usage prediction using the trained LSTM model
- `scripts/forecasting/plot_forecast.py` — Visualization of forecast results against historical usage data
- `requirements.txt` — Python dependencies for model training, inference server, and forecasting workflows
- `README.md` — Full guide covering the ML model development lifecycle, LLM pipeline stages, inference server setup, Nginx proxy, Open WebUI integration, and LSTM forecasting

```
models/
├── 📄 requirements.txt                  Python dependencies
├── 📄 README.md                         ML models, LLM pipeline, and inference server guide
└── 📂 scripts/
    ├── 📂 llm_from_scratch/
    │   ├── model.py                     GPT-style transformer architecture
    │   ├── tokenizer.py                 Tokenizer for pre-training and inference
    │   ├── train.py                     Pre-training script
    │   ├── text_classifier.py           Fine-tuning as a text classifier
    │   ├── chatbot.py                   Interactive chatbot with the pre-trained model
    │   ├── api_server.py                FastAPI inference server with REST endpoints
    │   └── requirements.txt             Dependencies for LLM training and inference
    ├── 📂 nginx/
    │   └── nginx.conf                   Nginx reverse proxy with token-based authorization
    ├── 📂 logging/
    │   └── log_analyzer.py              Request logging and CSV dataset generation
    └── 📂 forecasting/
        ├── prepare_data.py              Data preparation and normalization for LSTM
        ├── train_lstm.py                LSTM model training for load forecasting
        ├── forecast.py                  Next-day usage prediction
        └── plot_forecast.py             Forecast visualization against historical data
```

---

### 📂 observability/

**Key Points:**
- Real-time monitoring and visualization of agent behavior
- Security analysis including malicious tool detection
- Interactive dashboards for agent activity tracking
- Performance and behavioral analytics

**Importance for Autonomous AI:**
Transparency and monitoring are critical for autonomous systems. This folder provides visibility into what agents are doing, why they make decisions, and how they interact with tools and other agents. It enables debugging, security auditing, and building trust in autonomous AI systems.

**Key Features:**
- Weather dashboard example demonstrating real-time agent visualization
- Malicious tool detection analysis for security
- Web-based interfaces for monitoring agent activities
- Test frameworks for validation

### 📂 orchestration/

**Key Points:**
- Workflow orchestration and task coordination
- Complex multi-step agent operations
- Blog agent examples showing content generation workflows
- Testing infrastructure for orchestrated systems

**Importance for Autonomous AI:**
Autonomous agents often need to execute complex, multi-step workflows that involve coordination between multiple components. This folder provides patterns for orchestrating these workflows, managing dependencies, handling failures, and ensuring tasks complete successfully. It's essential for building production-grade autonomous systems that can handle real-world complexity.

**Key Features:**
- Blog content generation agents demonstrating end-to-end workflows
- Task coordination and dependency management
- Test suite for workflow validation
- Quickstart guides for rapid deployment

### 📂 retrieval/

**Key Points:**
- Open-source vector databases including Qdrant, Chroma, FAISS, Weaviate, and Milvus
- Vector search algorithms, similarity metrics, and indexing strategies (HNSW, IVF, scalar quantization)
- Retrieval-Augmented Generation (RAG) pipeline integrating vector search with large language models
- Hybrid search combining dense vector retrieval with keyword-based (BM25) search
- Local RAG with Microsoft Foundry Local for on-device LLM inference without cloud dependencies

**Importance for Autonomous AI:**
Autonomous agents operating on large knowledge bases cannot load entire document corpora into a context window. Vector databases solve this by converting documents into dense numerical embeddings and enabling sub-millisecond approximate nearest-neighbor search across millions of vectors. Retrieval-Augmented Generation grounds LLM responses in factual, up-to-date information by retrieving only the most semantically relevant passages before generating an answer, reducing hallucinations and enabling agents to work with private or domain-specific knowledge without retraining the model. For autonomous systems, RAG is a foundational pattern that decouples knowledge storage from model weights, allowing agents to reason over dynamic, evolving datasets at inference time.

**Key Features:**
- Overview of open-source vector databases with trade-off analysis for production selection
- End-to-end RAG pipeline example using Qdrant and Microsoft Foundry Local running locally on-device
- Coverage of distance metrics (cosine similarity, dot product, Euclidean) and approximate nearest-neighbor index types
- Quantization techniques for reducing memory footprint while preserving retrieval accuracy
- Hybrid search patterns merging dense embeddings with sparse keyword retrieval
- Document ingestion workflow from raw text to stored vectors with embedding model integration
- Vibe Coding workflow demonstrating AI-assisted development with a vector database backend
- Foundry Local integration for privacy-preserving, offline-capable RAG without cloud API calls

### 📂 workflows/

**Key Points:**
- SDLC integration with Vibe Coding methodology for AI-assisted development
- Foundry Local framework for running AI models entirely on local devices
- Natural language-based code generation through conversational AI agents
- Practical scripts including prompt assistant for optimized vibe coding workflows
- Virtual environment setup and management for isolated Python development

**Importance for Autonomous AI:**
The software development life cycle is being fundamentally transformed by autonomous AI agents that generate code through natural language descriptions, automate routine tasks, and provide intelligent assistance throughout development workflows. This folder demonstrates how traditional SDLC frameworks integrate with AI-assisted programming tools like GitHub Copilot, Claude Code, and Foundry Local to enable developers to act as orchestrators rather than manual coders. Vibe Coding represents a paradigm shift from write-then-test to interact-and-verify, where developers describe intent and AI agents generate implementation. Understanding this integration is essential for building modern development workflows that leverage autonomous agents while maintaining code quality, security, and maintainability standards.

**Key Features:**
- SDLC documentation covering planning, design, implementation, testing, deployment, and maintenance with AI agents
- Foundry Local setup guides for Linux environments with VS Code integration
- Prompt assistant tool for generating optimized vibe coding prompts using locally-running models
- Virtual environment management scripts for isolated Python development
- Project structure templates and best practices for AI-assisted workflows
- Integration patterns for autonomous coding agents within traditional SDLC phases
- Offline-capable AI development with complete data privacy and zero cloud dependencies
- AGENTS.md format documentation for guiding coding agents with project-specific conventions

### 📂 platform/

**Key Points:**
- Vendor-neutral, open-source AI engineering platform powered by [MLflow](https://github.com/mlflow/mlflow) for the full lifecycle of agents, LLM applications, and ML models
- Four interconnected pillars: **Tracing**, **Evaluation and Human Feedback**, **Prompt Versioning**, and **AI Governance**
- GenAI tracing with OpenTelemetry-compatible spans covering every LLM call, tool invocation, and agent step
- Built-in and custom LLM judges with 70+ quality scorers for automated evaluation pipelines
- Prompt Registry for versioned prompt management with lineage to evaluation results and traces
- AI Gateway providing centralized LLM routing, cost control, and content guardrails across providers
- Integrates with any agent framework including LangGraph, LangChain, OpenAI Agents SDK, Claude Agent SDK, Google ADK, Pydantic AI, AutoGen, CrewAI, and others
- Local deployment via Docker Compose and cloud deployment on Amazon SageMaker, Google Cloud Vertex AI, and Microsoft Azure Machine Learning
- Vibe coding workflow using VS Code and Claude to generate, trace, evaluate, and deploy MLflow applications through natural language prompts

**Importance for Autonomous AI:**
Autonomous agents and LLM applications fail silently — they produce plausible-sounding outputs that are incorrect, inconsistent, or subtly unsafe without raising exceptions or leaving obvious error traces. MLflow addresses this by providing the instrumentation, evaluation, and governance layer that transforms ad-hoc development into a measurable engineering discipline. Tracing captures every span of an agent's execution — LLM calls, tool invocations, retrieval steps, and intermediate reasoning — making failures visible and debuggable rather than opaque. Evaluation pipelines with LLM judges quantify quality across correctness, relevance, safety, and custom criteria, enabling regression testing between prompt versions or model updates. The Prompt Registry links prompt versions directly to their evaluation results, providing the kind of reproducibility and lineage tracking that production systems require. The AI Gateway centralizes access control and cost management across all LLM providers, which is essential when multiple autonomous agents operate concurrently against shared infrastructure. Together, these capabilities close the gap between prototype and production, giving teams the confidence to deploy autonomous agents at scale.

**Key Components:**
- `scripts/tracking/train_and_track.py` — Experiment tracking script logging parameters, metrics, and model artifacts for scikit-learn training runs
- `scripts/agents/langchain_agent_trace.py` — MLflow automatic tracing for a LangChain agent including tool calls and LLM input/output spans
- `scripts/agents/rag_pipeline_trace.py` — Manual span instrumentation for a RAG pipeline covering retrieval, prompt construction, and generation steps
- `scripts/evaluation/evaluate_qa.py` — GenAI evaluation pipeline using built-in Correctness and RelevanceToQuery scorers via `mlflow.genai.evaluate()`
- `scripts/evaluation/prompt_registry.py` — Prompt versioning workflow linking stored prompts to evaluation metrics and trace lineage
- `scripts/deployment/register_model.py` — Model Registry workflow for packaging, versioning, and stage-transitioning ML models
- `scripts/deployment/deploy_sagemaker.py` — Cloud deployment of a registered MLflow model to an Amazon SageMaker endpoint
- `docker/Dockerfile` — Container image definition for the MLflow tracking server
- `docker/docker-compose.yml` — Local deployment stack running the MLflow tracking server with a persistent artifact store
- `README.md` — Full tutorial covering MLflow pillars, vibe coding prompts, SDLC integration, Docker setup, and cloud deployment guides

```
platform/
├── 📄 README.md                         Full MLflow tutorial and deployment guide
├── 📂 docker/
│   ├── Dockerfile                       MLflow tracking server container image
│   └── docker-compose.yml               Local deployment stack with artifact store
└── 📂 scripts/
    ├── 📂 tracking/
    │   └── train_and_track.py           Experiment tracking with parameters and metrics
    ├── 📂 agents/
    │   ├── langchain_agent_trace.py     Automatic tracing for LangChain agents
    │   └── rag_pipeline_trace.py        Manual span tracing for RAG pipelines
    ├── 📂 evaluation/
    │   ├── evaluate_qa.py               GenAI evaluation with built-in LLM judges
    │   └── prompt_registry.py           Prompt versioning and evaluation lineage
    └── 📂 deployment/
        ├── register_model.py            Model Registry versioning and stage transitions
        └── deploy_sagemaker.py          Cloud deployment to Amazon SageMaker
```

---

### 📂 optimization/

**Key Points:**
- Techniques for reducing model size and inference cost without sacrificing quality
- Quantization (INT4/INT8), pruning, knowledge distillation, and low-rank factorization
- Parameter-Efficient Fine-Tuning (PEFT) and LoRA for task-specific adaptation with minimal compute
- Prompt optimization and Retrieval-Augmented Generation (RAG) for accuracy improvements
- Cloud-based optimization on Microsoft Azure (Olive, Azure ML, Azure AI Search), AWS, and Google Cloud
- Local optimization workflows for on-device inference with reduced memory footprint

**Importance for Autonomous AI:**
Autonomous agents that rely on large language models face practical constraints in latency, memory, and cost that limit real-world deployment. Optimization techniques address these constraints directly: quantization shrinks model VRAM requirements from 28 GB (FP32) to as little as 3.5 GB (INT4), enabling deployment on consumer hardware; LoRA and PEFT allow agents to be fine-tuned on domain-specific data without retraining billions of parameters; and prompt optimization combined with RAG ensures agents produce accurate, grounded responses without inflating model size. For autonomous systems operating at scale, the difference between an unoptimized and an optimized model can determine whether deployment is economically and technically feasible.

**Key Components:**
- `quantization_example.py` - 4-bit and 8-bit quantization using Hugging Face and BitsAndBytes
- `fast_finetune.py` - Parameter-efficient fine-tuning with LoRA and PEFT
- `prompt_optimization.py` - Few-shot prompt engineering for improved model accuracy
- `rag_example.py` - Local RAG pipeline integrating vector search with LLM inference
- `azure_rag_search.py` - RAG optimization with Azure AI Search
- `azure_ml_submit.py` - Cloud-based training and optimization jobs on Azure ML

### 📂 reasoning/

**Key Points:**
- Chain-of-Thought (CoT) prompting for structured, multi-step problem solving
- Reinforcement Learning (RL) training approaches including GRPO for reasoning behavior
- Test-Time Computation (TTC) scaling to improve answer quality at inference
- Tree of Thoughts and self-correction for exploring and refining solution paths
- Large Reasoning Model (LRM) training: SFT + RL, pure RL, and distillation pipelines
- Local deployment of reasoning models with Ollama and Open WebUI
- Agentic reasoning with tool use via the ReAct (Reason + Act) loop

**Importance for Autonomous AI:**
Reasoning is the cognitive core of autonomous agents. Standard LLMs retrieve statistically likely answers; reasoning models calculate answers by working through intermediate steps, enabling reliable performance on mathematics, coding, planning, and complex logic tasks. Chain-of-Thought transforms model outputs from pattern retrieval into verifiable thought traces, allowing agents to self-correct and explore alternative solution paths before committing to an answer. Reinforcement learning on reasoning — through algorithms like GRPO and Process Reward Models (PRMs) — teaches models to improve their reasoning chains based on outcome quality, not just next-token prediction. For autonomous systems, robust reasoning capability directly determines the complexity of tasks an agent can reliably execute without human supervision.

**Key Components:**
- `prompts/` - System prompt templates and prompt strategy configurations for reasoning models
- `scripts/` - Training scripts including Unsloth GRPO fine-tuning and Ollama deployment workflows

### 📂 security/

**Key Points:**
- A security framework for AI agents in production
- Protection against prompt injection, data poisoning, and adversarial attacks
- Layered security approach combining traditional and AI-specific methods
- Identity and access management with least privilege principles
- Guardrails for input/output inspection and content safety

**Importance for Autonomous AI:**
Autonomous AI agents execute code, manage files, and access multiple applications with minimal human oversight. This autonomy introduces significant security risks including prompt injection attacks, memory poisoning, and unauthorized access to sensitive systems. A robust security framework is essential to ensure agents operate within safe boundaries, protect against malicious manipulation, and maintain trust in production environments. Without proper security measures, autonomous agents can be exploited to bypass controls, leak sensitive data, or perform unintended actions.

**Key Features:**
- Identity and Access Management (IAM) with service accounts and short-lived credentials
- Input/output inspection using guard models like Llama Guard and ShieldGemma
- Network and infrastructure security with isolated networks and traffic inspection
- Data and secret security using cloud secrets managers and encryption
- Monitoring and governance with real-time agent activity tracking
- Protection against prompt injections, indirect attacks, and AI memory poisoning
- Practical implementations with Ollama and Google Agent Development Kit
- Use cases for securing Amazon Bedrock Agents and GKE deployments

---

### 📂 training/

**Key Points:**
- Full training lifecycle for both **Large Language Models (LLMs)** and classical **Machine Learning (ML)** models
- LLM pipeline organised into two broad phases: **pre-training** (building general language capabilities from massive unlabelled corpora) and **post-training** (aligning and adapting those capabilities for instruction-following and human preferences)
- Multi-stage pre-training covering core pre-training, context-length extension, and annealing — the approach used by Llama 3.1, Qwen 2, and Apple's AFM
- **Distributed pre-training of large language models** across multi-GPU clusters using NativeDDP, DeepSpeed ZeRO (stages 1–3), and PyTorch FSDP — parallelism strategies, hardware requirements, communication protocols (NCCL / UCX), and fault-tolerant checkpointing are covered in depth
- Ready-to-run distributed pre-training scripts in `scripts/` for DDP, DeepSpeed ZeRO-2/3, and FSDP, with YAML configs for GPT-2 (124 M) and GPT-2 XL (1.5 B)
- **[LLaMA-Factory](https://llamafactory.readthedocs.io/en/latest/index.html)** — open-source unified framework for pre-training and fine-tuning 100+ LLMs through a zero-code CLI and Web UI; integrates DeepSpeed ZeRO, FSDP, and Flash Attention 2; supports SLURM-based multi-node scheduling; used in production by Amazon, NVIDIA, and Alibaba Cloud
- Post-training alignment pipeline: Supervised Fine-Tuning (SFT) → Reward Modelling → Policy Optimisation (RLHF/PPO, DPO, ORPO, KTO)
- Parameter-Efficient Fine-Tuning (PEFT) with LoRA for task-specific adaptation at 0.1–6 % of total parameters
- ML model training for classification and regression with PyTorch and scikit-learn, including a reusable data pipeline
- Distributed training strategies: Data Parallelism (DDP/FSDP), Tensor Parallelism, Pipeline Parallelism, and 3D Parallelism

**Importance for Autonomous AI:**
Training is the process that creates the intelligence powering every autonomous agent in this repository. Pre-training on trillions of tokens gives a model its factual knowledge, language patterns, and commonsense reasoning — the raw capability that all downstream use depends on. Post-training alignment transforms that raw capability into reliable, instruction-following behaviour: an unaligned base model will complete sequences unpredictably, whereas an aligned model consistently answers helpfully, honestly, and safely. For autonomous systems, alignment is not optional — agents that execute code, access APIs, and operate with minimal human oversight must refuse dangerous instructions, acknowledge uncertainty, and produce outputs that match human expectations across edge cases no rule-based filter can anticipate. Fine-tuning with LoRA allows teams to encode domain-specific behaviour directly into model weights at low compute cost, while the ML training scripts provide the classical forecasting and classification capabilities that complement LLMs in production pipelines.

**Key Components:**
- `llm/pretrain.py` — GPT-style causal language model pre-training from scratch; supports multi-GPU via `torchrun` and `accelerate`; configurable architecture via JSON config
- `llm/finetune.py` — Instruction fine-tuning with LoRA/PEFT adapters in the Alpaca prompt format; masks prompt tokens from the loss; saves adapter-only checkpoints
- `llm/requirements.txt` — LLM training dependencies: `torch`, `transformers`, `datasets`, `peft`, `accelerate`, `wandb`, `tensorboard`
- `llm/setup_venv.sh` — Virtual environment setup for the LLM training scripts
- `ml/data_pipeline.py` — Reusable data preparation module; handles missing values, categorical encoding, `StandardScaler` on train only, stratified splits, and `DataLoader` objects
- `ml/train_classifier.py` — MLP/CNN classifier training with `OneCycleLR` scheduling and `CrossEntropyLoss`; saves best-validation-accuracy checkpoint
- `ml/train_regression.py` — MLP regressor training with `CosineAnnealingLR`; evaluates with RMSE, MAE, and R²; saves best-validation-RMSE checkpoint
- `ml/requirements.txt` — ML training dependencies: `torch`, `scikit-learn`, `pandas`, `numpy`, `datasets`, `wandb`, `tensorboard`
- `ml/setup_venv.sh` — Virtual environment setup for the ML training scripts
- `scripts/setup.sh` — Installs LLaMA-Factory, DeepSpeed, and creates `scripts/.venv`; verifies Python 3.11+, GPU count, and CUDA
- `scripts/run_ddp.sh` — NativeDDP pre-training of GPT-2 (124 M) on 2 GPUs via `FORCE_TORCHRUN=1`
- `scripts/run_deepspeed_z2.sh` — DeepSpeed ZeRO-2 pre-training of GPT-2 XL (1.5 B); shards optimizer states and gradients
- `scripts/run_deepspeed_z2_offload.sh` — ZeRO-2 with optimizer CPU offload; reduces VRAM ~6 GB at ~20–40 % throughput cost
- `scripts/run_deepspeed_z3_offload.sh` — ZeRO-3 with full CPU offload (params + optimizer); maximises model size at throughput cost
- `scripts/run_fsdp.sh` — FSDP FULL_SHARD pre-training via `accelerate launch`; equivalent to ZeRO-3 with PyTorch-native sharding
- `scripts/configs/` — Ten YAML and JSON configs covering DDP, DeepSpeed ZeRO-2/3 (with and without CPU offload), and FSDP strategies
- `README.md` — Full guide covering the LLM training pipeline, alignment techniques, dataset selection, distributed training strategies, adaptation methods, and ML model lifecycle

**LLM Training Stages:**

| Stage | Phase | Description |
|---|---|---|
| Core pre-training | Pre-training | Next-token prediction on a broad, diverse corpus; acquires factual knowledge and language patterns |
| Continued pre-training | Pre-training | Context-length extension to 32k–128k tokens using long-document data |
| Annealing | Pre-training | Short final phase on a small, high-quality curated mix to sharpen benchmark performance |
| Supervised Fine-Tuning (SFT) | Post-training | Trains on instruction-response pairs; teaches the model to follow instructions and adopt an assistant format |
| Reward Modelling | Post-training | Trains a secondary model on human preference rankings to output a scalar quality score |
| Policy Optimisation | Post-training | RLHF/PPO, DPO, ORPO, or KTO to align the policy with the reward signal |
| LoRA / PEFT | Adaptation | Adapter-only fine-tuning for task-specific domains at ~0.1–6 % of parameters |

**Alignment of Large Language Models:**

Post-training is often called **alignment** because it is the phase that closes the gap between a base model's raw sequence-completion capability and behaviour that is *helpful, honest, and harmless* (HHH) — the three canonical alignment goals.

The alignment pipeline has three stages:

1. **Supervised Fine-Tuning (SFT).** The base model is trained on curated instruction-response pairs using cross-entropy loss applied only to the response tokens. SFT teaches the model the basic structure of an assistant interaction — how to answer a prompt rather than continue it — and is the foundation for all subsequent alignment steps. Training data combines human-annotated examples with synthetically generated pairs produced by stronger, already-aligned models.

2. **Reward Modelling (RM).** Human annotators (or AI proxies) rank multiple model responses to the same prompt. These preference comparisons train a Reward Model to output a scalar score representing how much a human would prefer a given response, capturing nuanced judgements about tone, accuracy, safety, and helpfulness that rules cannot encode.

3. **Policy Optimisation.** The LLM's generation policy is updated to consistently produce responses that score highly on the Reward Model. Four principal algorithms are used in production today:

   - **RLHF with PPO** — The classical approach. The model generates responses, the Reward Model scores them, and weights are updated via Proximal Policy Optimisation. A KL-divergence penalty prevents the model from drifting too far from the pre-trained distribution, preserving fluency. High compute cost and training instability have motivated simpler alternatives.
   - **Direct Preference Optimisation (DPO)** — Treats the LLM itself as an implicit reward function and optimises directly on preference pairs (chosen vs. rejected) without training a separate Reward Model. More stable, cheaper, and widely adopted — Llama 3.1, Qwen 2, and most recent open-weight models use SFT + iterative DPO.
   - **ORPO (Odds Ratio Preference Optimisation)** — Combines SFT and preference optimisation into a single training step using an odds-ratio penalty on rejected responses, eliminating the need for a separate SFT phase and reducing pipeline complexity.
   - **KTO (Kahneman-Tversky Optimisation)** — Inspired by behavioural economics prospect theory. Optimises from binary *good / bad* labels rather than pairwise comparisons, making data collection cheaper and the method applicable when paired examples are unavailable.

Without alignment, a capable base model is an unreliable sequence completer that may produce harmful, factually incorrect, or unhelpful outputs with no mechanism for refusal or self-correction. Alignment is what transforms raw model capability into the trustworthy, goal-directed behaviour that production autonomous agents require.

```
training/
├── 📄 .gitignore                        Excludes virtual environments, model outputs, and build artefacts
├── 📄 README.md                         LLM and ML training guide with alignment deep dive
├── 📂 llm/
│   ├── requirements.txt                 Python dependencies for LLM pre-training and fine-tuning
│   ├── setup_venv.sh                    Creates and activates a virtual environment
│   ├── pretrain.py                      GPT-style causal LM pre-training from scratch
│   └── finetune.py                      Instruction fine-tuning with LoRA / PEFT adapters
├── 📂 ml/
│   ├── requirements.txt                 Python dependencies for ML training
│   ├── setup_venv.sh                    Creates and activates a virtual environment
│   ├── data_pipeline.py                 Load, clean, split, scale, and wrap data as PyTorch DataLoaders
│   ├── train_classifier.py              MLP / CNN classifier training with TensorBoard logging
│   └── train_regression.py              MLP regressor training (RMSE, MAE, R²)
└── 📂 scripts/
    ├── setup.sh                         Install LLaMA-Factory, DeepSpeed, and venv
    ├── run_ddp.sh                       NativeDDP pre-training on 2 GPUs (GPT-2 124 M)
    ├── run_deepspeed_z2.sh              DeepSpeed ZeRO-2 pre-training (GPT-2 XL 1.5 B)
    ├── run_deepspeed_z2_offload.sh      DeepSpeed ZeRO-2 + CPU offload pre-training
    ├── run_deepspeed_z3_offload.sh      DeepSpeed ZeRO-3 + full CPU offload pre-training
    ├── run_fsdp.sh                      FSDP FULL_SHARD pre-training via accelerate
    └── configs/                         YAML and JSON configs for all five distributed strategies
```

---

## Getting Started

Each folder contains its own `README.md` with specific setup instructions and usage examples. Start with the `autonomous/` folder for core concepts, then explore `collaboration/` and `orchestration/` to see how agents work together and execute complex workflows.

## Documentation

- `autonomous/docs/` - Architecture, setup guides, and tutorials
- Individual folder READMEs for specific component documentation
- `QUICKSTART.md` files for quick project setup

## License

MIT License - see [LICENSE](LICENSE) file for details.

