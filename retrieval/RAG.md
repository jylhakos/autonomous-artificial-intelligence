# Retrieval-Augmented Generation (RAG)

A tutorial to Retrieval-Augmented Generation, vector databases, chunking strategies, and evaluation metrics for building production-ready RAG systems.

## Table of Contents

- [What is Retrieval-Augmented Generation?](#what-is-retrieval-augmented-generation)
- [How Vector Search Works](#how-vector-search-works)
  - [Chunking](#chunking)
  - [Why Chunking Matters](#why-chunking-matters)
  - [Why is Chunking Important for RAG?](#why-is-chunking-important-for-rag)
  - [What is Chunking?](#what-is-chunking)
- [Chunking Strategies](#chunking-strategies)
  - [Fixed-Size Chunking](#fixed-size-chunking)
  - [Sentence-Level Chunking](#sentence-level-chunking)
  - [Semantic Chunking](#semantic-chunking)
  - [Document-Based Chunking](#document-based-chunking)
  - [Recursive Chunking](#recursive-chunking)
- [Optimizing for Retrieval Accuracy](#optimizing-for-retrieval-accuracy)
- [Vector Database Integration](#vector-database-integration)
  - [LlamaIndex Integration](#llamaindex-integration)
  - [LangChain Integration](#langchain-integration)
  - [Weaviate Integration](#weaviate-integration)
  - [Pinecone Integration](#pinecone-integration)
- [Chunk Sizes and Optimization](#chunk-sizes-and-optimization)
- [Azure AI Integration](#azure-ai-integration)
- [Evaluation](#evaluation)
  - [What is Evaluation?](#what-is-evaluation)
  - [Key Metrics](#key-metrics)
  - [Retrieval Metrics](#retrieval-metrics)
  - [Generation Metrics](#generation-metrics)
- [Code Examples](#code-examples)
- [References and Resources](#references-and-resources)
- [Getting Started](#getting-started)

---

## What is Retrieval-Augmented Generation?

**Retrieval-Augmented Generation (RAG)** is an architectural pattern that enhances Large Language Models (LLMs) by combining them with external knowledge retrieval. Instead of relying solely on the model's training data, RAG systems:

1. **Retrieve** relevant information from a vector database based on the user's query
2. **Augment** the LLM prompt with this retrieved context
3. **Generate** accurate, grounded responses using both the query and retrieved information

### Benefits of RAG:
- **Up-to-date information**: Access current data without retraining the model
- **Reduced hallucinations**: Responses grounded in actual documents
- **Domain specialization**: Leverage private/proprietary knowledge bases
- **Cost-effective**: No need for expensive model fine-tuning
- **Transparency**: Can trace answers back to source documents

---

## How Vector Search Works

Vector search forms the foundation of RAG systems. It works by:

1. **Converting text to embeddings**: Transform documents and queries into high-dimensional vectors
2. **Storing vectors**: Index embeddings in a vector database (e.g., Weaviate, Pinecone)
3. **Similarity search**: Compare query embeddings with document embeddings using cosine similarity or other distance metrics
4. **Retrieving top-k results**: Return the most semantically similar chunks

### Chunking

Chunking is a critical preprocessing step in the RAG pipeline that directly impacts retrieval quality and answer accuracy.

### Why Chunking Matters

Chunking matters because:
- **Embedding quality**: Embedding models perform best on coherent, focused text segments
- **Search precision**: Smaller chunks enable more precise semantic matching
- **Context windows**: LLMs have token limits; chunks must fit within these constraints
- **Cost optimization**: Smaller, targeted chunks reduce token usage and API costs

### Why is Chunking Important for RAG?

**Getting chunking right is one of the most important decisions in building your RAG pipeline.** How you split your documents affects:

1. **Your system's ability to find relevant information**: Poorly chunked data leads to irrelevant retrievals
2. **Answer accuracy**: Chunks that split context mid-thought produce incomplete or confusing answers
3. **Retrieval efficiency**: Optimal chunk sizes balance specificity with sufficient context

### What is Chunking?

**Chunking is the process of breaking down large documents into smaller, manageable pieces called chunks.** This is a first step when preparing data for use with Large Language Models (LLMs).

For example, a 50-page PDF might be split into 200 chunks of ~512 tokens each, ensuring:
- Each chunk contains a complete thought or topic
- Chunks have some overlap to preserve context at boundaries
- Chunk size fits within embedding model limits (typically 512-2048 tokens)

---

## Chunking Strategies

Choosing the right chunking strategy depends on your data type, structure, and retrieval needs.

### Fixed-Size Chunking

**Splits text by a set number of characters or tokens** (e.g., 512 tokens per chunk with 50-100 tokens of overlap).

**Best for**: Basic documents with uniform structure  
**Pros**: Simple, predictable, fast  
**Cons**: May split sentences or paragraphs mid-thought

**Reference**: [Pinecone - Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)

> Pinecone highlights the necessity of choosing the right chunk size and strategies (like sliding windows with overlap) to ensure embedding models can easily map data and the resulting vectors contain the needed information for search.

### Sentence-Level Chunking

**Breaks text into individual sentences or groups of sentences**, keeping complete logical thoughts together.

**Best for**: Documents where sentence boundaries are important (articles, papers)  
**Pros**: Preserves semantic units  
**Cons**: May create very small or very large chunks depending on sentence length

### Semantic Chunking

**Uses AI to identify when the actual meaning or topic of the text shifts**, creating chunks based on natural paragraph or theme breaks rather than arbitrary character counts.

**Best for**: Complex documents with varied topics  
**Pros**: Highest quality semantic coherence  
**Cons**: Slower, requires embeddings for every potential split point

**Reference**: [Weaviate - Chunking Strategies](https://weaviate.io/blog/chunking-strategies-for-rag)

> Weaviate emphasizes advanced methods like **Semantic Chunking** and **Late Chunking**, which group text by meaning rather than arbitrary token counts to ensure search precision without sacrificing the surrounding context.

### Document-Based Chunking

**Respects the native layout and structure of the document**, using HTML tags, JSON keys, or Markdown headers to define chunk boundaries.

**Best for**: Structured content (web pages, codebases, API docs)  
**Pros**: Preserves logical document structure  
**Cons**: Requires parsing logic specific to each format

### Recursive Chunking

**Breaks text down hierarchically using a list of delimiters** (e.g., paragraphs, then sentences, then words) until chunks fit the size limit.

**Best for**: Varied, unstructured text  
**Pros**: Balances structure preservation with size constraints  
**Cons**: More complex implementation

---

## Optimizing for Retrieval Accuracy

**The first step is making sure your system can find the right information in your vector database.** Vector search does this by comparing user queries with the embeddings of your chunks.

### Key Optimization Techniques:

1. **Chunk overlap**: Use 10-20% overlap between chunks to prevent context loss at boundaries
2. **Metadata enrichment**: Add source, page number, section headers as metadata for filtering
3. **Hybrid search**: Combine vector search with keyword search for better precision
4. **Query transformation**: Rephrase or expand queries before retrieval
5. **Re-ranking**: Use a cross-encoder to re-score retrieved chunks

**References**:
- [Microsoft - RAG Chunking Phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase)
- [Microsoft - RAG Preparation Phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-preparation-phase)

---

## Vector Database Integration

### LlamaIndex Integration

**To integrate LlamaIndex with a vector database**, you initialize the database's specific VectorStore class, wrap it inside a LlamaIndex StorageContext, and pass that context into a VectorStoreIndex.

**To use chunking with LlamaIndex and a Weaviate vector database**, you split documents into smaller structural nodes using LlamaIndex's ingestion parameters or specialized Node Parsers before passing them into the WeaviateVectorStore.

**Data Indexing**: Once the data is loaded, LlamaIndex offers the ability to index this data with a wide variety of data structures and storage integration options (including Weaviate).

**Retrieval Engine**: Once your data is ingested/stored, LlamaIndex provides the tools to define an advanced retrieval/query "engine" over your data. Our retriever constructs allow you to retrieve data from your knowledge base given an input prompt.

**References**:
- [LlamaIndex - Weaviate Auto Retriever](https://developers.llamaindex.ai/python/framework/integrations/vector_stores/weaviateindex_auto_retriever/)
- [LlamaIndex - Vector Store Integrations](https://developers.llamaindex.ai/python/framework/community/integrations/vector_stores/)
- [LlamaIndex - Basic Optimization Strategies](https://developers.llamaindex.ai/python/framework/optimizing/basic_strategies/basic_strategies/)

### LangChain Integration

The most common way to chunk documents is using the **LangChain library**, which can handle both vector databases.

See [chunking_langchain.py](chunking_langchain.py) for complete implementation.

### Weaviate Integration

**Weaviate's Python client** allows you to pass in raw text and will auto-vectorize via configured modules (like text2vec-openai), but it still requires the text to be chunked pre-ingestion.

See implementation examples in:
- [chunking_langchain.py](chunking_langchain.py)
- [chunking_llamaindex.py](chunking_llamaindex.py)

### Pinecone Integration

**Pinecone does not split text internally**; you must chunk your text, embed each chunk individually, and store them as separate records with IDs that connect back to the parent document.

**Reference**: [Pinecone - Data Modeling](https://docs.pinecone.io/guides/index-data/data-modeling)

See [chunking_langchain.py](chunking_langchain.py) for ingestion implementation.

---

## Chunk Sizes and Optimization

### Recommended Chunk Sizes:

- **Small chunks (128-256 tokens)**: High precision, may lack context
- **Medium chunks (512-1024 tokens)**: Balanced approach, most common
- **Large chunks (1024-2048 tokens)**: More context, may reduce relevance

### Choosing Your Chunking Strategy:

Select a chunking method based on your data type and retrieval needs:

| Strategy | Chunk Size | Overlap | Best For |
|----------|-----------|---------|----------|
| **Fixed-size** | 512 tokens | 50-100 tokens | Basic documents |
| **Recursive** | 1000 chars | 200 chars | Unstructured text |
| **Document-based** | Variable | N/A | Structured content (HTML, Markdown, code) |
| **Semantic/Agentic** | Variable | N/A | Complex reasoning, multi-topic texts |

**Reference**: [LlamaIndex - Chunk Sizes](https://developers.llamaindex.ai/python/framework/optimizing/basic_strategies/basic_strategies/)

---

## Azure AI Integration

**To implement chunking using Microsoft Azure AI Foundry** (for models and embeddings), LlamaIndex (for orchestration), and Weaviate (as the vector database), you use LlamaIndex's SentenceSplitter to break documents into segments. These chunks are embedded via Azure OpenAI and ingested into a Weaviate collection.

### Tutorial:

**Build a RAG pipeline using Azure Files with LlamaIndex and Weaviate**:  
[Microsoft Tutorial - LlamaIndex + Weaviate](https://learn.microsoft.com/en-us/azure/storage/files/artificial-intelligence/retrieval-augmented-generation/open-source-frameworks/tutorials/llamaindex-weaviate/tutorial-llamaindex-weaviate)

**Develop applications with LlamaIndex and Microsoft Foundry**:  
[Microsoft Foundry - LlamaIndex Integration](https://learn.microsoft.com/en-us/azure/foundry-classic/how-to/develop/llama-index)

See [chunking_llamaindex.py](chunking_llamaindex.py) for full Azure AI Foundry integration example.

---

## Evaluation

### What is Evaluation?

**Evaluation** is the systematic measurement of your RAG system's performance across retrieval quality, context relevance, and answer accuracy. Without proper evaluation, you cannot:

- Determine if chunking changes improve or degrade performance
- Compare different embedding models or retrieval strategies
- Identify failure modes (e.g., retrieving irrelevant chunks, hallucinated answers)
- Optimize for production quality and cost

### Key Metrics

RAG evaluation spans two critical phases:

#### Retrieval Metrics

Measure **how well your system finds relevant information**:

- **Precision@K**: What proportion of top-K retrieved chunks are relevant?
- **Recall@K**: What proportion of all relevant chunks appear in top-K?
- **Mean Reciprocal Rank (MRR)**: How quickly does the first relevant chunk appear?
- **NDCG@K**: Normalized measure accounting for graded relevance and ranking position

#### Generation Metrics

Measure **the quality of generated answers**:

- **Context Relevance**: Are retrieved chunks actually relevant to the query?
- **Answer Faithfulness**: Is the answer grounded in retrieved context (no hallucinations)?
- **Answer Relevance**: Does the answer actually address the user's question?
- **Latency**: How long does end-to-end retrieval + generation take?

### Evaluation Implementation

See [rag_evaluation.py](rag_evaluation.py) for complete implementation of all evaluation metrics.

**Example evaluation workflow**:

```python
from rag_evaluation import evaluate_retrieval, RAGMetrics

# Evaluate retrieval quality
retrieval_metrics = evaluate_retrieval(
    retrieved_docs=["doc1", "doc3", "doc5"],
    relevant_docs=["doc1", "doc2"],
    relevance_scores={"doc1": 1.0, "doc2": 0.8, "doc3": 0.0},
    k=5
)

print(f"Precision@5: {retrieval_metrics.precision_at_k:.3f}")
print(f"NDCG@5: {retrieval_metrics.ndcg_at_k:.3f}")
```

---

## Code Examples

This repository includes complete implementations:

### 1. [chunking_langchain.py](chunking_langchain.py)

Demonstrates chunking with **LangChain** for both **Weaviate** and **Pinecone**:
- Fixed-size chunking
- Recursive chunking
- Ingestion pipelines for both vector databases

### 2. [chunking_llamaindex.py](chunking_llamaindex.py)

Demonstrates chunking with **LlamaIndex** and **Azure AI Foundry**:
- Sentence-level chunking
- Semantic chunking
- Azure OpenAI integration
- Weaviate vector store setup
- Query engine construction

### 3. [rag_evaluation.py](rag_evaluation.py)

An evaluation framework:
- Retrieval metrics (Precision, Recall, MRR, NDCG)
- Context relevance evaluation
- Answer faithfulness checking
- Answer relevance scoring

### Usage Example

```python
# Using LangChain with Weaviate
from chunking_langchain import fixed_size_chunking, ingest_to_weaviate
from langchain_community.document_loaders import TextLoader

# Load and chunk documents
loader = TextLoader("your_document.txt")
docs = loader.load()
chunks = fixed_size_chunking(docs, chunk_size=512, chunk_overlap=50)

# Ingest into Weaviate
ingest_to_weaviate(chunks, collection_name="MyDocs")
```

```python
# Using LlamaIndex with Azure AI and Weaviate
from chunking_llamaindex import setup_azure_models, semantic_chunking, create_weaviate_index
from llama_index.core import SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Setup Azure models
embed_model, llm = setup_azure_models()

# Apply semantic chunking
nodes = semantic_chunking(documents, embed_model)

# Create searchable index
index = create_weaviate_index(nodes, embed_model, "MyCollection")
query_engine = index.as_query_engine(llm=llm)

# Query
response = query_engine.query("What are the key concepts?")
```

---

## References and Resources

### Microsoft Azure
- [RAG Chunking Phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase)
- [RAG Preparation Phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-preparation-phase)
- [Tutorial: LlamaIndex + Weaviate with Azure Files](https://learn.microsoft.com/en-us/azure/storage/files/artificial-intelligence/retrieval-augmented-generation/open-source-frameworks/tutorials/llamaindex-weaviate/tutorial-llamaindex-weaviate)
- [LlamaIndex with Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry-classic/how-to/develop/llama-index)

### Vector Databases & Frameworks
- [Weaviate - Chunking Strategies for RAG](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Pinecone - Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [Pinecone - Data Modeling Guide](https://docs.pinecone.io/guides/index-data/data-modeling)
- [LlamaIndex - Weaviate Auto Retriever](https://developers.llamaindex.ai/python/framework/integrations/vector_stores/weaviateindex_auto_retriever/)
- [LlamaIndex - Vector Store Integrations](https://developers.llamaindex.ai/python/framework/community/integrations/vector_stores/)
- [LlamaIndex - Optimization Strategies](https://developers.llamaindex.ai/python/framework/optimizing/basic_strategies/basic_strategies/)

### Key Concepts
- **Chunking**: Breaking documents into smaller, semantically coherent pieces
- **Embeddings**: Vector representations of text that capture semantic meaning
- **Vector Search**: Finding similar documents using distance metrics in embedding space
- **RAG Pipeline**: Retrieve → Augment → Generate workflow
- **Retrieval Accuracy**: Quality measure of finding relevant information
- **Context Relevance**: Quality measure of retrieved chunks
- **Answer Faithfulness**: Measure of grounding in source documents

---

## Getting Started

### Prerequisites

```bash
pip install langchain langchain-community langchain-openai
pip install llama-index llama-index-vector-stores-weaviate llama-index-embeddings-azure-openai
pip install weaviate-client pinecone-client
pip install numpy
```

### Environment Variables

For Azure OpenAI:
```bash
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_EMBEDDING_DEPLOYMENT="text-embedding-ada-002"
export AZURE_LLM_DEPLOYMENT="gpt-4"
```

For Pinecone:
```bash
export PINECONE_API_KEY="your-key"
```

### Quick Start

1. **Prepare your data**: Place documents in a `data/` directory
2. **Choose a chunking strategy**: See [Chunking Strategies](#chunking-strategies)
3. **Select a vector database**: Weaviate or Pinecone
4. **Run the pipeline**: Use provided scripts
5. **Evaluate**: Measure retrieval and generation quality

---

## License

MIT License

---
