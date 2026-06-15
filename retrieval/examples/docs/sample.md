# Vector Database and RAG Quick Reference

This document provides a comprehensive reference on vector databases, indexing algorithms, chunking strategies, retrieval-augmented generation, and evaluation metrics. It is included as sample content for the RAG example script.

## Table of Contents

- [What is a Vector Database?](#what-is-a-vector-database)
- [What is Retrieval-Augmented Generation?](#what-is-retrieval-augmented-generation)
- [What is Chunking?](#what-is-chunking)
- [Why is Chunking Important for RAG?](#why-is-chunking-important-for-rag)
- [Chunking Strategies](#chunking-strategies)
- [Key Indexing Algorithms](#key-indexing-algorithms)
- [Distance Metrics](#distance-metrics)
- [RAG Evaluation and Metrics](#rag-evaluation-and-metrics)
- [Open Source Vector Databases](#open-source-vector-databases)
- [Microsoft Foundry Local](#microsoft-foundry-local)
- [Implementation Examples](#implementation-examples)
- [References](#references)

---

## What is a Vector Database?

A vector database is a specialized database system that stores data as high-dimensional
numerical vectors called embeddings. Unlike traditional relational databases that use exact
keyword matching, vector databases enable semantic similarity search — finding data based on
meaning rather than precise terms.

Each record stores three components: a unique ID, the numerical vector produced by an
embedding model, and optional metadata such as source filename or document category.

---

## What is Retrieval-Augmented Generation?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances Large Language Model (LLM) responses by grounding them in external knowledge retrieved from a vector database. Instead of relying solely on the model's training data, RAG retrieves relevant documents or chunks from a knowledge base and injects them as context into the LLM prompt before generating a response.

### Why RAG Matters

RAG addresses several critical limitations:
- **Reduces Hallucinations**: Provides factual context from trusted sources
- **Up-to-Date Information**: Access to current data beyond the model's training cutoff
- **Domain Expertise**: Grounds LLMs in proprietary documents and knowledge bases
- **Source Attribution**: Retrieved chunks can be traced back to source documents
- **Cost Efficiency**: More economical than fine-tuning large models

### The RAG Pipeline

**Offline Ingestion**:
1. Load documents from various sources
2. **Chunk** documents into smaller segments
3. Embed chunks using an embedding model
4. Store vectors in a vector database

**Online Retrieval**:
1. Embed user query
2. Retrieve top-k similar chunks
3. Inject chunks as context into LLM prompt
4. Generate grounded response

**Reference**: [Microsoft Azure RAG Guide](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-preparation-phase)

---

## What is Chunking?

**Chunking** is the process of breaking down large documents into smaller, manageable pieces called **chunks**. This is a crucial first step when preparing data for use with Large Language Models (LLMs) in RAG applications.

### Why Chunking is Necessary

1. **Embedding Model Context Limits**: Most embedding models have a maximum input size (e.g., 512 or 1024 tokens)
2. **Semantic Precision**: Smaller, focused chunks capture clear semantic meaning better than large, mixed-topic documents
3. **LLM Context Window Efficiency**: Passing only relevant chunks is faster and more cost-effective
4. **Retrieval Accuracy**: Well-chunked content produces more precise search results

### The Chunking Challenge

LLMs have limited context windows, meaning they can only process a certain amount of text at once. If there is too much text within the context window, important details are lost, resulting in incomplete or inaccurate answers.

Chunking solves this by creating smaller, focused pieces of content that an LLM can use to answer queries without getting lost in irrelevant information.

**Reference**: [Weaviate: What is Chunking?](https://weaviate.io/blog/chunking-strategies-for-rag)

---

## Why is Chunking Important for RAG?

**Getting chunking right is one of the most important decisions in building your RAG pipeline.** How you split your documents affects your system's ability to find relevant information and give accurate answers.

### 1. Optimizing for Retrieval Accuracy

The first step is making sure your system can find the right information in your vector database. Vector search does this by comparing user queries with the embeddings of your chunks.

**Chunks that are too large**:
- Often mix multiple ideas together
- Create a noisy, "averaged" embedding
- Don't clearly represent any single topic
- Make it hard for vector retrieval to find relevant context

**Chunks that are small and focused**:
- Capture one clear idea
- Result in precise embeddings
- Encode all the nuanced parts of the content
- Make it easier for your system to find the right information

For **AI agents**, this retrieval step effectively becomes a form of **long-term memory**, where well-formed chunks determine what the agent can recall later.

### 2. Preserving Context for Generation

After your system finds the best chunks, they're passed to the LLM. This is where context quality determines the quality of the outputted response.

**Chunks that are too small**:
- Fail to provide sufficient context
- Like reading a single sentence from the middle of a research paper
- Even humans would struggle to understand without more context

**Chunks that are too large**:
- Create attention dilution
- LLM performance degrades with longer context inputs
- "Lost in the middle" effect: models struggle with information buried in long contexts
- Increased likelihood of hallucinating responses

### The Chunking Sweet Spot

You want to preserve the author's "train of thought" while creating chunks that are small enough for precise retrieval but complete enough to give the LLM full context.

**When you get this balance right**:
- ✅ **Improves Retrieval Quality**: Focused, semantically complete chunks enable precise context retrieval
- ✅ **Manages LLM Context Window**: Only relevant data gets passed to the LLM
- ✅ **Reduces Hallucinations**: Providing small, highly relevant chunks grounds the model in factual data
- ✅ **Enhances Efficiency**: Processing smaller chunks is faster and more cost-effective

**References**:
- [Weaviate: Why is Chunking Important?](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Microsoft Azure: RAG Chunking Phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase)

---

## Chunking Strategies

There are several approaches to chunking, each with different trade-offs:

### 1. Fixed-Size Chunking

**What it is**: Splits text by a set number of characters or tokens (e.g., 512 tokens per chunk).

**When to use**: Quick prototyping, unstructured text, or when document structure is inconsistent. Best for basic documents and getting started.

**Best practices**:
- Use **10-20% overlap** between chunks to preserve context at boundaries
- Measure in **tokens** rather than characters for consistency with embedding models
- Start with **512 tokens** and adjust based on your embedding model's capacity

**Implementation**:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Target max characters per chunk
    chunk_overlap=50     # Overlap between chunks to retain context
)

chunks = text_splitter.split_documents(docs)
```

**Reference**: [Pinecone: Fixed-Size Chunking](https://www.pinecone.io/learn/chunking-strategies/)

### 2. Sentence-Level Chunking

**What it is**: Breaks text into individual sentences or groups of sentences, keeping complete logical thoughts together.

**When to use**: Text with clear sentence boundaries, prose documents, articles, or essays.

**Tools**: spaCy sentence tokenizer, NLTK sentence tokenizer, LangChain recursive text splitter.

### 3. Semantic Chunking

**What it is**: Uses AI to identify when the actual meaning or topic of the text shifts, creating chunks based on natural paragraph or theme breaks rather than arbitrary character counts.

**When to use**: Dense, unstructured text like academic papers, legal documents, or complex narratives where semantic boundaries matter more than structural ones.

**How it works**:
1. Break text into sentences
2. Generate embeddings for each sentence
3. Calculate similarity between consecutive sentences
4. Create chunk boundaries where similarity drops (topic changes)

This approach ensures each chunk contains a self-contained idea or topic, significantly improving retrieval quality.

**Reference**: [Weaviate: Semantic Chunking](https://weaviate.io/blog/chunking-strategies-for-rag#semantic-chunking)

### 4. Document-Based Chunking

**What it is**: Respects the native layout and structure of the document, using HTML tags, JSON keys, or Markdown headers to define chunk boundaries.

**When to use**: Highly structured documents like Markdown files, HTML pages, code files, or JSON documents.

**Examples**:
- **Markdown**: Split by headings (`#`, `##`, `###`)
- **HTML**: Split by tags (`<p>`, `<div>`, `<section>`)
- **Code**: Split by functions or classes (`def`, `class` in Python)

**Tools**: LangChain document-specific splitters (MarkdownTextSplitter, HTMLTextSplitter).

### 5. Recursive Chunking

**What it is**: Breaks text down hierarchically using a list of delimiters (e.g., paragraphs, then sentences) until the chunk fits the size limit.

**When to use**: Varied, unstructured text with natural separators. Good middle ground between fixed-size and semantic approaches.

**Default separators**: `["\n\n", "\n", " ", ""]` (paragraphs → sentences → words)

### Chunking Strategy Comparison

| **Strategy** | **Complexity** | **When to Use** | **Tools** |
|-------------|---------------|----------------|-----------|
| **Fixed-size** | Low | Quick prototyping, basic documents | LangChain RecursiveCharacterTextSplitter |
| **Sentence-level** | Low | Prose with clear sentence boundaries | spaCy, NLTK, LangChain |
| **Recursive** | Medium | Unstructured text with natural separators | LangChain RecursiveCharacterTextSplitter |
| **Document-based** | Medium | Structured documents (HTML, Markdown, code) | LangChain MarkdownTextSplitter, HTMLTextSplitter |
| **Semantic** | High | Complex reasoning, multi-topic texts | Custom implementation with embeddings |

**Key Recommendations**:
- Start with **fixed-size chunking with 10-20% overlap** (512 tokens, 50-100 token overlap)
- Experiment with different chunk sizes and measure retrieval quality
- Use document structure when available (headers, paragraphs, sections)
- Consider semantic chunking for complex, high-value documents

**References**:
- [Weaviate: Chunking Strategies](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Pinecone: Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [Microsoft Azure: Chunking Approaches](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase)

---

## Key Indexing Algorithms

Vector databases organize high-dimensional embeddings using approximate nearest-neighbor
(ANN) algorithms to avoid scanning every stored vector for each query.

HNSW (Hierarchical Navigable Small World) builds a multi-layered directed graph. Upper
layers contain coarse, long-range connections for rapid navigation; lower layers contain
dense, precise connections for accurate retrieval. Search time grows logarithmically with
dataset size. HNSW is the default index in Qdrant, Weaviate, and Chroma.

IVF (Inverted File Index) partitions vectors into k clusters using k-means. At query time,
only the clusters whose centroids are nearest to the query vector are scanned. IVF variants
such as IVF-PQ combine cluster partitioning with product quantization for memory-efficient
retrieval at billion-scale. FAISS and Milvus use IVF extensively.

Flat indexing performs an exhaustive brute-force scan of all stored vectors. It guarantees
exact nearest neighbours but has linear time complexity, making it suitable only for small
collections or when 100% recall is mandatory.

## Distance Metrics

Cosine similarity measures the angle between two vectors and is widely used for text
embeddings. Values range from -1 (opposite) to 1 (identical direction). Euclidean distance
(L2) measures the straight-line distance between two points in vector space. Dot product
combines magnitude and direction and is common in recommendation systems.

## Retrieval-Augmented Generation (RAG)

RAG grounds language model responses in external knowledge. The offline ingestion pipeline
converts documents into text chunks, embeds each chunk using an embedding model, and stores
the vectors in a vector database. At query time, the user's question is embedded, the
database retrieves the most semantically similar chunks, and those chunks are injected as
context into the language model's prompt before generation. This reduces hallucinations and
allows the model to cite specific sources.

## Open Source Vector Databases

Qdrant is an open-source Rust-based vector database with HNSW indexing, metadata filtering
within the graph traversal, and support for scalar, product, and binary quantization. It is
well-suited for production workloads requiring real-time updates and low-latency search.

Chroma is a lightweight Apache 2.0 database optimised for developer prototyping and
conversational AI memory. It uses HNSW by default and requires minimal configuration.

FAISS from Meta FAIR is a C++ library with Python bindings providing a broad range of index
types including flat, IVF, HNSW, and their quantized variants. GPU acceleration is
supported. FAISS is embedded directly in applications rather than run as a server.

Weaviate is a cloud-native vector database with integrated vectorization, generative RAG
queries, hybrid search, and a knowledge-graph-style data model.

## Microsoft Foundry Local

Microsoft Foundry Local is an on-device AI runtime that downloads, manages, and serves
language models entirely on a local device without requiring cloud connectivity or external
API keys. It exposes an OpenAI-compatible REST API at http://localhost:5272/v1, allowing any
application written against the OpenAI API to switch to a local model by changing only the
base URL. Models are executed via ONNX Runtime and support CPU, GPU, and NPU backends.

Foundry Local is particularly useful in air-gapped environments, on the factory floor, and
in any scenario where data must not leave the local device. Combined with a local vector
database such as Qdrant, it enables a fully offline RAG pipeline with no external
dependencies after the initial model download.
