# Graph Retrieval-Augmented Generation (GraphRAG)

This document presents concepts, architectural components, practical use cases and implementation technologies associated with Graph retrieval-augmented generation (GraphRAG). By leveraging knowledge graphs alongside large language models, GraphRAG extends retrieval-augmented generation approaches with structured representations of entities and relationships, facilitating improved contextual retrieval and complex reasoning over interconnected information.

## Table of Contents

- [Introduction](#introduction)
  - [What is GraphRAG?](#what-is-graphrag)
  - [How GraphRAG Works?](#how-graphrag-works)
  - [How GraphRAG improves retrieval?](#how-graphrag-improves-retrieval)
  - [Why RAG misses connected information?](#why-rag-misses-connected-information)
- [Use Cases](#use-cases)
  - [When GraphRAG is useful?](#when-graphrag-is-useful)
  - [Example: Customer Support Assistant](#example-customer-support-assistant)
- [Architectures](#architectures)
  - [GraphRAG Architecture](#graphrag-architecture)
- [Agentic AI and GraphRAG](#agentic-ai-and-graphrag)
  - [Why make GraphRAG agentic?](#why-make-graphrag-agentic)
  - [GraphRAG vs Agentic GraphRAG](#graphrag-vs-agentic-graphrag)
  - [Building an Agentic GraphRAG](#building-an-agentic-graphrag)
- [Building Samples](#building-samples)
  - [Prerequisites](#prerequisites)
  - [Setup Virtual Environment](#setup-virtual-environment)
  - [Dataset Generation](#dataset-generation)
  - [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
  - [GraphRAG](#graphrag)
  - [Agentic AI with RAG](#agentic-ai-with-rag)
  - [Agentic AI with GraphRAG](#agentic-ai-with-graphrag)
- [Project Structure](#project-structure)
- [References](#references)


## Introduction

RAG improves the reliability of GenAI components by ensuring that LLM answers are based only on accurate information from existing knowledge sources.

RAG excels at localized, keyword-adjacent fact-finding, but it fails to see the bigger picture. GraphRAG extends retrieval-augmented generation approaches to overcome this limitation by incorporating structured representations of entities and relationships.

RAG retrieves chunks of text from documents using vector similarity. GraphRAG retrieves entities and relationships from a graph structure, allowing the LLM to reason over connected information rather than isolated text passages.

### What is GraphRAG?

GraphRAG is used for complex datasets where traditional semantic search fails.

### How GraphRAG Works?

Building GraphRAG pipeline involves two major phases:

Knowledge Graph (KG) Construction (using an LLM to extract entities/relationships from text and storing them) and Graph Retrieval & Generation (querying the database using vector math combined with graph steps).

1. Indexing (Graph Creation):

Extraction: The system scans your raw document data. An LLM acts as an extraction mechanism to identity key entities (e.g., "Project Alpha", "Sarah") and their relationships (e.g., "Sarah manages Project Alpha").

Clustering & Summarization: Frameworks like Microsoft's GraphRAG partition these interconnected nodes into "communities" using algorithms like Leiden. The LLM then pre-writes summaries for these broader groups.

2. Querying (Retrieval & Generation):

Local Search: For entity-specific queries (e.g., "What did Sarah work on?"), the system pinpoints the "Sarah" node and pulls all adjacent data connected via its edges.

Global Search: For holistic, macro questions (e.g., "What were our biggest logistical risks in Q3?"), the retriever bypasses isolated text chunks and searches across the pre-compiled community summaries to form an aggregate answer.

Generation: The structured context is added to the prompt, forcing the LLM to output a grounded, accurate response.

### How GraphRAG improves retrieval?

GraphRAG solves "Multi-Hop" problems: If a query requires connecting relevant details across document A, B, and C (e.g., "Find all employees who used Python in a project managed by John"), GraphRAG seamlessly steps across edges to chain the facts together.

RAG indexes unstructured documents using dense embeddings. Graph-based retrieval instead constructs an entity network, a structured semantic layer that's especially useful in domains where the relationships among concepts carry the answer.

**RAG vs. GraphRAG**

| Feature | Baseline (Vector) RAG | GraphRAG |
|-------------|---------------|---------------|
| Data format | Flat chunks of text | Interconnected nodes and edges |
| Search approach | Math-based vector similarity | Graph traversal and relationship tracking |
| Query strengths | Specific "fact-lookup" queries | "Multi-hop" reasoning and contextual summaries |
| Explainability | Low (black-box vector math) | High (traceable relationship paths) |

**GraphRAG benefits over RAG**

1. Multi-hop reasoning

Questions such as:

"Which customers are affected by suppliers connected to Factory A?"

require several relationship hops.

Graph traversal naturally supports this.

2. Better context precision

Instead of retrieving large document chunks, GraphRAG can retrieve only the entities and relationships relevant to the query.

3. Explainability

The system can show:

```
      Customer C
            ← receives
      Product B
            ← produced by
      Factory A

```

### Why RAG misses connected information?

RAG treats every chunk as an isolated object, with no native way to represent how chunks relate to each other. In a typical pipeline, your app converts documents into chunks, embeds each chunk as a high-dimensional vector, and stores those vectors in an index. At query time, the app embeds the user's question, retrieves the closest vectors by distance, and passes those chunks to the LLM as context.

## Use Cases

### When GraphRAG is useful?

GraphRAG is most valuable when information contains many **interconnected** entities and relationships.

| Use Case | Why GraphRAG Helps |
|-------------|---------------|
| Enterprise knowledge management | Connect employees, projects, documents, products, and customers |
| Customer support | Link products, components, error codes, manuals, and troubleshooting procedures |
| Healthcare | Connect diseases, symptoms, treatments, medications, and research papers |
| Financial analysis | Model companies, subsidiaries, executives, markets, and transactions |
| Cybersecurity | Represent hosts, users, vulnerabilities, attacks, and alerts |
| Scientific research | Link authors, publications, methods, datasets, and findings |
| Supply chain management | Track suppliers, parts, warehouses, shipments, and customers |

4. Reduced hallucinations

The model receives structured facts from the graph rather than relying entirely on semantic similarity.

### Example: Customer Support Assistant

Suppose a company manufactures elevators.

**RAG**

The system retrieves:

```
Document A: Motor overheating issue
Document B: Sensor calibration procedure
Document C: Maintenance schedule

```

The LLM must infer relationships from separate chunks.

**GraphRAG**

The **graph** explicitly stores:


```
      Elevator A
      ├── contains → Motor X
      ├── contains → Sensor Y
      └── maintained_by → Procedure Z

      Motor X
      └── causes → Error E102

      Error E102
      └── resolved_by → Cooling Procedure P

```

When a user asks:

"Why does Elevator A show Error E102?"

The retrieval system can traverse:

```
      Elevator A
      → Motor X
      → Error E102
      → Cooling Procedure P

```
The LLM receives the relevant chain of evidence instead of unrelated document chunks.

Given your interests in AI agents, RAG, vector databases, and enterprise deployments, GraphRAG would be particularly useful for:

Technical support chatbots that connect products, components, manuals, and error codes.

## Architectures

For many enterprise AI systems, a common architecture:


```
                Documents
                    │
                    ├─► Vector Database (semantic search)
                    │
                    └─► Knowledge Graph (GraphRAG)
                              │
                              ▼
                             LLM

```

### GraphRAG Architecture


```
                Documents
                     │
                     ▼
          Entity & Relation Extraction
                     │
                     ▼
               Knowledge Graph
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   Graph Traversal       Vector Search
          │                     │
          └──────────┬──────────┘
                     ▼
               Retrieved Context
                     ▼
                    LLM
                     ▼
                  Response

```

## Agentic AI and GraphRAG

Agentic AI utilizes Graph retrieval-augmented generation to move beyond simple question-answering by using the structural connections and entities in a knowledge graph to dynamically plan, trigger, and execute complex workflows.

### Why make GraphRAG agentic?

While GraphRAG is great, it has one flaw: rigidity.

Numerous RAG systems rely solely on opens in **vector search** over text embeddings (numerical vector representations) for information retrieval.

Instead of matching exact keywords, [vector search](https://weaviate.io/blog/vector-search-explained) assesses the conceptual meaning and contextual intent of the query.

Vector search returns similar items based on their semantic meaning rather than exact term matches, by comparison GraphRAG retrieves information based on explicit relationships and structure.

Instead of just matching similar concepts, GraphRAG maps out specific entities and traces exactly how they connect to one another.

GraphRAG extracts "entities" (e.g., people, products, places) from your documents and builds a connected web of data.

Agents, are simply decision-making systems that use LLMs to interact with external data sources.

At its core, an agent is simply an LLM call with a specific context. It takes your input and reasons through what it knows, and it interacts with external tools or databases to retrieve the most relevant answers.

For example, instead of always relying on vector search, an agent might use a graph traversal algorithm like BFS or PageRank to find the most relevant nodes in a knowledge graph.

When a user submits a complex query, an orchestrator agent breaks the query down into smaller tasks. It then decides which tools to use – querying a vector database, calling an external API, checking a CRM, or executing a Python script.

Flexibility: Agents aren’t tied to one approach—they select the best tool or algorithm for the task.

Autonomy: Once set up, the system operates independently, requiring minimal human intervention.

Error handling: Feedback loops allow agents to dynamically retry, diagnose issues, and recover from failures.

GraphRAG systems hard-code pipelines for data retrieval. This is great for simple use cases but limiting for complex, evolving scenarios.

### GraphRAG vs Agentic GraphRAG

| Feature | GraphRAG | Agentic GraphRAG |
|-------------|---------------|---------------|
| Retrieval mechanism | Graph traversal (Nodes and Edges) | Autonomous multi-step orchestration |
| Implementation | Requires ontology and data structuring | Requires agent orchestration and tool integration |
| Query complexity  | Relational, multi-hop queries | Open-ended problem solving |
| Vulnerability | Garbage In, Garbage Out | High latency, infinite loops without guardrails |
| Use Cases | Static data structuring, relationship queries, and analytical tasks | Multi-step reasoning, dynamic troubleshooting, and tool-driven workflows |

### Building an Agentic GraphRAG

Building an Agentic GraphRAG system involves creating a pipeline of agents, each responsible for a specific step in the process.

Assess your Query complexity:

Are your users asking straightforward questions like, “What is the company travel policy?”

-> Start with RAG (enhanced with basic reranking or hybrid search).

Are users asking, “Which enterprise clients are impacted by the outage of Server X?”

-> You need the structural mapping of Graph RAG.

Are users asking, “Analyze Q3 market trends, cross-reference our top 5 competitor filings, and draft a strategic response plan?”

-> You must implement Agentic RAG.

Evaluate Latency and Cost constraints:

If you are building a real-time customer-facing chatbot, the high latency and token costs of Agentic RAG can quickly become excessive.

In these scenarios, highly optimized RAG or Graph RAG architectures are preferable.

1. Multi-Hop reasoning for Complex decisions:

Standard vector search can only match text chunks based on semantic similarity; it cannot string together a sequence of connected facts. GraphRAG allows the agent to traverse relationships (e.g., Employee -> Project -> Client -> Vendor).

The Action: The agent can logically deduce dependencies before taking an action. For instance, before approving an invoice, the agent traces the graph to verify the corresponding vendor, cross-reference the active contract, and check for pending deliverables.

2. Precise Tool selection & Parameter passing:

Agents have access to specialized tools (e.g., an email API, a database query tool, or a payment endpoint). Instead of guessing which tool to use, the agent uses the knowledge graph to map entities to system actions.

The integration of GraphRAG transforms an agent's capability into an agentic workflow through several key mechanisms:

1. Question classification

The first agent analyzes the user query and determines its type. For example:

Retrieval questions involve direct lookups, such as “What is a dollar?”

Structured questions explore relationships, like “Is dollar related to stock market trends?”

Global queries analyze trends such as “What are the top 10 most important nodes?”

2. Tool selection

Based on the query type, another agent selects the best tool to retrieve the required data. This might include vector search, PageRank, BFS (breadth-first search), DFS (depth-first search), or a database schema query.

3. Execution

The selected tool processes the query, retrieving relevant data or results. Agents can adapt tools to match the query’s specific needs. For example, a PageRank query may return the top 10 nodes for a broad query or a single node for a specific question.

4. Feedback loops

If a tool fails (e.g., due to a misconfigured query or missing data), the agent diagnoses the issue, retries, or switches to an alternative tool.

5. Response

The processed data is returned to the user as a meaningful answer.

## Building Samples

A step-by-step guide to setting up a local RAG, Agentic AI, and GraphRAG environment and pipeline using Ollama, Python, LangGraph and Neo4j with the Iris dataset.

### Prerequisites

Before starting, ensure you have the following installed:
- Python 3.8 or higher
- Ollama (for running local LLMs)
- Docker (for Neo4j)
- Git

### Setup Virtual Environment

Create and activate a Python virtual environment to isolate dependencies:

```bash
cd /path/to/graph
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate  # On Windows
```

### Dataset Generation

The Iris dataset is used throughout these examples to demonstrate RAG and GraphRAG capabilities.

#### Generate Synthetic Dataset

Use the generation script to create a text document with semantic descriptions of the Iris dataset:

**Install Dependencies:**

```bash
pip install pandas scikit-learn langchain-text-splitters
```

**Run the Generation Script:**

```bash
python scripts/dataset/generate_dataset_for_vector_database.py
```

This will create `document.txt` in the root directory containing 150 semantic descriptions of Iris specimens with morphological measurements (sepal length, sepal width, petal length, petal width) and species classifications (setosa, versicolor, virginica).

The script transforms raw tabular data into natural language suitable for vector embeddings, converting numerical measurements into descriptive sentences like: "This biological specimen belongs to the genus Iris, specifically classified under the species 'setosa'. Morphological measurements indicate a sepal length of 5.1 cm..."

### RAG (Retrieval-Augmented Generation)

To build a local retrieval-augmented generation (RAG) system, you can use Ollama to run Llama 3.2, ChromaDB as your local vector database, and LangChain to orchestrate the pipeline.

RAG agent relies on two different models: a reasoning or generation model (does the deciding, grading, rewriting and answering) and an embedding model (turns your documents and queries into vectors).

**Install Ollama and Models**

Download and install Ollama for your operating system:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama server:

```bash
ollama serve
```

Pull the required models:

```bash
# Reasoning model (the agent)
ollama pull llama3.2

# Embedding model (the retriever)
ollama pull nomic-embed-text
```

**Install Python Dependencies**

```bash
pip install langchain langchain-ollama langchain-community langchain-chroma chromadb
```

**RAG Script (rag_app.py)**

Location: `scripts/rag/rag_app.py`

This script demonstrates a basic RAG pipeline:
1. Loads the Iris dataset document (document.txt)
2. Splits it into chunks
3. Creates embeddings using nomic-embed-text
4. Stores vectors in ChromaDB
5. Retrieves relevant context for queries
6. Generates answers using Llama 3.2

**Run the RAG Script:**

```bash
python scripts/rag/rag_app.py
```

The script will query: "What are the morphological characteristics of Iris setosa?" and return an answer based on retrieved context from the Iris dataset.

**Streamlit Chat Interface (app.py)**

Location: `scripts/rag/app.py`

For an interactive web interface, use the Streamlit app which provides a chat-like experience.

**Install Streamlit:**

```bash
pip install streamlit
```

**Launch the Streamlit App:**

```bash
streamlit run scripts/rag/app.py
```

Your browser will open to http://localhost:8501 where you can interact with the Iris dataset through a conversational interface.

**Example Questions:**
- "What are the characteristics of Iris setosa?"
- "What is the typical petal length for Iris virginica?"
- "How do the three Iris species differ in their morphological measurements?"

### GraphRAG

To create a local GraphRAG (Graph-based retrieval-augmented generation) pipeline, you will use Ollama to host the Llama 3.2 LLM and an embedding model, LangChain to orchestrate the retrieval, and a graph database like Neo4j or Microsoft's GraphRAG SDK to structure the knowledge graph.

To build a [retrieval agent using LangGraph and Neo4j (GraphRAG)](https://neo4j.com/blog/developer/neo4j-graphrag-workflow-langchain-langgraph/), you construct an agentic system that routes queries between structured Cypher query generation and unstructured vector similarity search, combining the results inside a stateful workflow.

Steps:

Query analysis and routing:

The user’s request is first analyzed and classified, allowing the system to route it to the appropriate workflow node. Depending on the query, the system may proceed to the next step (research plan generation), prompt the user for clarification, or respond immediately if the request is out of scope.

Research plan generation:

The system constructs a detailed, step-by-step research plan tailored to the complexity of the user’s query. This plan outlines the specific actions required to fulfill the request.

Research graph execution: For each step in the research plan, a dedicated subgraph is invoked. The system generates Cypher queries via LLMs, targeting the Neo4j knowledge graph. Relevant nodes and relationships are retrieved using a hybrid approach that combines semantic search and structured graph queries, ensuring both breadth and precision in the results.

Answer generation: Leveraging the retrieved graph data, the system synthesizes a response using an LLM, integrating information from multiple sources as needed.

To get started, create a project space and python virtual environment to install graphrag.

**Activate virtual environment**

source .venv/bin/activate

**Install the required dependencies**

pip install --upgrade pip

pip install langchain langchain-community langchain-ollama chromadb sentence-transformers ollama

**Run Neo4j with APOC Plugins via Docker**

Neo4j needs the APOC (Awesome Procedures on Cypher) plugin enabled so LangChain can map and structure the knowledge graph correctly.

docker run \
    -d \
    --name neo4j-graphrag \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password123 \
    -e NEO4J_PLUGINS='["apoc"]' \
    -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
    neo4j:5.20.0

Open http://localhost:7474 in your browser to inspect your graph. Log in with username neo4j and password password123.

Ensure you have the official Neo4j driver and the experimental LangChain graph components installed inside your virtual environment

pip install neo4j langchain-experimental


**Install GraphRAG**

Note: If you use Microsoft's GraphRAG package, you will also install it via pip install graphrag

python -m pip install graphrag

**Initialize GraphRAG**

Extract Entities and Build the Knowledge Graph

GraphRAG and LangChain require specific packages to communicate with your local Ollama models.

When prompted, specify the default chat and embedding models you would like to use in your config.

You need to parse your documents into entities (nodes) and their relationships (edges).

You can do this automatically using LangChain's built-in LLMGraphTransformer and a local graph store like Neo4j, which can be easily spun up on your virtual Linux machine using Docker.

**Set up workspace variables**

Create the Retrieval Chain and Chat

**Crate Chat Script**

**Execute the Script**

python3 graph_rag_chat.py

**Query**

Once your knowledge graph and vector representations are established, you can use LangChain to bind them with your Chat and Llama3.2. The model can perform both vector search and graph traversal to answer user queries.

Now let's ask some questions using this dataset.

### Agentic AI with RAG

Agents can dynamically choose tools, incorporate complex reasoning, and adapt their analysis approach based on the situation at hand. ReAct Agents are an agentic architecture that use “reasoning and acting” via tool calling supported LLMs. Agentic AI with RAG extends the concept of RAG paradigm by integrating retrieval mechanisms with the language understanding and reasoning capabilities of large language models.

To build a [retrieval agent](https://docs.langchain.com/oss/python/langchain/retrieval) using LangGraph involves running a model on Ollama, storing documents in a local vector database, and using an agentic framework like LangGraph to decide whether to search documents or rewrite queries.

Creating a local [RAG agent](https://docs.langchain.com/oss/python/langgraph/agentic-rag) involves setting up Ollama as your LLM and embedding engine, ChromaDB as your vector store, and LangGraph to manage the routing logic.

Implementing a local Agentic RAG system requires setting up your local environment, loading your documents into ChromaDB, and building a State Graph workflow in LangGraph.

You can also use [Langflow's](https://www.langflow.org/templates/use-langflow-to-build-local-rag-pipeline-with-ollama-and-chromadb) drag-and-drop UI to build this without coding.

**Ollama:**

Ensure Ollama is running and download the Llama3.2 and Embedding models.

ollama pull llama3.2
ollama pull nomic-embed-text

**ChromaDB:**

Run ChromaDB locally in a Docker container with persistent storage:

docker run -d -p 8000:8000 -v chroma-data:/chroma/data -e IS_PERSISTENT=TRUE chromadb/chroma

**Document Ingestion**

Before the agent can query, your text documents must be embedded and stored.

Install the dependencies:

pip install langchain langchain-ollama langchain-chroma langgraph

Use this Python script in ./scripts/agentic/rag folder to ingest your documents.

**Agent**

LangGraph allows the agent to decide when to search the local database, when to answer, or how to rewrite the query.

**The agentic RAG loop: retrieve, reason, act**

See the source code in ./scripts/agentic/rag folder.

### Agentic AI with GraphRAG

Implementing a Graph Query Agent involves creating a state machine in LangGraph where the local LLM acts as the reasoning engine to write and execute Cypher queries against a Neo4j database.

LangGraph lets you define custom workflows as [graphs](https://neo4j.com/labs/genai-ecosystem/genai-frameworks/langgraph/)

Smaller local LLMs hosted on Ollama can struggle with native tool-calling features, the most reliable method is to provide the agent with a single, highly constrained Cypher execution tool.

The latest Neo4j Cypher procedures now support Ollama models via configuring the baseURL (and using Ollama’s OpenAI-compatible endpoint).

**Prerequisites:**

Activate Virtual Environment

source .venv/bin/activate

Ollama: Ensure you have Ollama installed and a capable model (e.g., llama3.1:8b) pulled via ollama 

pull llama3.1:8b.

Neo4j: Have a Neo4j instance running locally or via Neo4j Aura.

Libraries: Install the necessary Python packages:

pip install langgraph langchain-ollama langchain-community neo4j-driver

Make sure your local Ollama instance has the model pulled:

ollama pull llama3.2

**Implementation:**

To integrate both Cypher Graph Queries and Vector RAG Search into a single cohesive LangGraph agent, we will build a hybrid agent using local Ollama (Llama 3.2) and Neo4j.

This allows the agent to dynamically choose whether it needs to do a semantic text search (Vector) or a structured relationship search (Cypher) depending on the user's question.

1. Chunk, Embed, and Store document.txt in Neo4j

This script in scripts folder reads your text file, creates vector embeddings using Ollama, and stores them inside Neo4j's native vector index.

2. Define State and Connect to Databases

First, define the agent state (using MessagesState) and set up your Neo4j connection using the Neo4j Python Driver.

3.  Define Tools (Vector & Cypher)

We create two separate tools decorated with @tool. Llama 3.2 will look at their names and descriptions to decide which one to invoke.

Tool calling via langchain-ollama:

Because llm was initialized using ChatOllama(model="llama3.2") from the langchain-ollama package, LangChain automatically passes the JSON schema of your @tool functions to Ollama.

When you run agent_executor.invoke(), Llama 3.2 uses its native tool-calling capabilities to generate a structured request specifying exactly which tool to run and with what query arguments. LangGraph's internal prebuilt edges catch that instruction and execute the code for you.

4. Build the LangGraph Retriever Agent

We will define a retriever tool from Neo4j and wrap it inside a LangGraph agent. Llama 3.2 supports tool calling natively via the langchain-ollama provider.

**Build the hybrid LangGraph Agent**

We combine both tools into a unified ReAct agent execution graph. We pass a state_modifier system prompt to help the smaller 3B Llama model route effectively.

5. Create the Graph Query Tool

Create a function that translates a text query into an exact Cypher string. Bind this to your LLM using LangChain's @tool decorator.

Defining a Retriever Tool from Neo4j

In Step 4 of the integrated code, the following lines explicitly convert your Neo4j vector database index into a tool:

```

# Turns the Neo4j index into a LangChain retriever object
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Wraps that retriever object inside a function decorated as a tool
@tool
def search_unstructured_text(query: str) -> str:
    """Useful for answering semantic questions..."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


```

Wrapping it inside a LangGraph Agent

In Step 4, the create_react_agent function takes that tool, packages it with your LLM, and compiles it into a structured LangGraph state machine:

```

tools = [search_unstructured_text, query_graph_relationships]

# This compiles the full LangGraph node-and-edge engine
agent_executor = create_react_agent(llm, tools, state_modifier=system_instructions)


```

6. Build the LangGraph Nodes and Edges

Using the ReAct framework, we define a node for the agent to reason, a node to execute tools, and conditional edges to loop until an answer is found.

7. Run the Agent

Invoke the agent with a specific question.

8. Test and Query

Now we can fire separate question variations at the exact same compiled graph executor to see it dynamically adjust its tooling strategy.

See the source code in ./scripts/agentic/graphrag/vector_search.py file.

See the source code in ./scripts/agentic/graphrag/structured_cypher_query.py file.

**Tips for Local LLMs:**

JSON Mode or Prompting: If your local model hallucinates or misses tool-calling syntax, instruct it to explicitly output responses in JSON format within your system prompt.

Schema Injection: For better Cypher generation accuracy, dynamically inject the Neo4j schema (node labels and relationship types) into your system prompt so the local LLM knows exactly what data structure to query.

See the source code in scripts/agentic/graphrag folder.

### Agentic AI with GraphRAG

Agentic AI with GraphRAG creates intelligent systems that can dynamically choose between vector search and graph traversal based on the query type. This hybrid approach combines the best of both worlds: semantic similarity search for unstructured text and structured relationship queries for graph data.

**Prerequisites**

Ensure you have:
- Ollama running with llama3.2 and nomic-embed-text models
- Neo4j running with APOC plugins (see GraphRAG section)
- Python dependencies installed

**Install Dependencies**

```bash
pip install langgraph langchain-ollama langchain-community neo4j-driver
```

**Step 1: Chunk, Embed, and Store in Neo4j (chunk_embed_and-populate.py)**

Location: `scripts/agentic/graphrag/chunk_embed_and-populate.py`

This script reads the Iris dataset document, creates vector embeddings using Ollama, and stores them inside Neo4j's native vector index.

**Run the script:**

```bash
python scripts/agentic/graphrag/chunk_embed_and-populate.py
```

**Step 2: Define Tools (tools.py)**

Location: `scripts/agentic/graphrag/tools.py`

This module defines two tools:
1. **search_unstructured_text**: For semantic queries about Iris morphological characteristics
2. **query_graph_relationships**: For structured queries about species relationships and node counts

The tools are decorated with `@tool` so Llama 3.2 can intelligently choose which one to invoke.

**Step 3: Build the Hybrid Agent (agents.py)**

Location: `scripts/agentic/graphrag/agents.py`

This script combines both tools into a unified ReAct agent that can:
- Analyze the query type
- Choose the appropriate tool (vector or graph)
- Execute the query
- Return structured results

**Step 4: Test Vector Search (vector_search.py)**

Location: `scripts/agentic/graphrag/vector_search.py`

Test semantic queries:

```bash
python scripts/agentic/graphrag/vector_search.py
```

Example query: "What are the main morphological characteristics of Iris virginica?"

**Step 5: Test Graph Queries (structured_cypher_query.py)**

Location: `scripts/agentic/graphrag/structured_cypher_query.py`

Test structural graph queries:

```bash
python scripts/agentic/graphrag/structured_cypher_query.py
```

Example query: "How many DocumentChunk nodes exist in the database and what species are represented?"

**How It Works**

1. **Query Classification**: The agent analyzes whether the question requires semantic search or graph traversal
2. **Tool Selection**: Based on query type, the agent chooses the appropriate tool
3. **Execution**: The selected tool processes the query against Neo4j
4. **Response**: Results are synthesized into an answer

**Example Use Cases**

- Semantic queries: "Describe the characteristics of Iris setosa"
- Graph queries: "Show all relationships between species and measurements"
- Hybrid queries: "Compare petal lengths across all three Iris species"

GraphRAG application involves generating Cypher query language with the LLM.

## Project Structure

```
graph/
├── README.md
├── document.txt (generated by dataset script)
├── scripts/
│   ├── dataset/
│   │   ├── generate_dataset_for_vector_database.py  # Generate Iris dataset
│   │   └── generate_summary_report.py
│   ├── rag/
│   │   ├── rag_app.py              # Basic RAG implementation
│   │   └── app.py                  # Streamlit chat interface for RAG
│   ├── graphrag/
│   │   └── graph_rag_chat.py       # GraphRAG with Neo4j
│   └── agentic/
│       ├── rag/
│       │   ├── ingest.py           # Ingest documents into ChromaDB
│       │   └── agent_with_rag.py   # Agentic RAG implementation
│       └── graphrag/
│           ├── chunk_embed_and-populate.py  # Populate Neo4j vector index
│           ├── tools.py            # Define vector and graph tools
│           ├── agents.py           # Build hybrid ReAct agent
│           ├── vector_search.py    # Test vector search queries
│           └── structured_cypher_query.py   # Test Cypher queries
└── venv/  # Python virtual environment
```

## References

What is GraphRAG? https://neo4j.com/blog/genai/what-is-graphrag/

Vector Search Explained https://weaviate.io/blog/vector-search-explained

Getting Started https://microsoft.github.io/graphrag/get_started/

The easiest way to build with open models https://ollama.com/

ChatOllama https://reference.langchain.com/python/langchain-ollama/chat_models/ChatOllama

Agentic AI use case https://docs.cloud.google.com/architecture/agentic-ai-multimodal-graph-rag-resource-orchestration

Build a custom RAG agent with LangGraph https://docs.langchain.com/oss/python/langgraph/agentic-rag

Build a RAG application with LangChain and Local LLMs powered by Ollama https://devblogs.microsoft.com/cosmosdb/build-a-rag-application-with-langchain-and-local-llms-powered-by-ollama/

**License**: MIT

**Last Updated**: June 26, 2026