# Vector Database Quick Reference

This document provides a concise reference on vector databases, indexing algorithms, and
retrieval-augmented generation. It is included as sample content for the RAG example script.

## What is a Vector Database?

A vector database is a specialized database system that stores data as high-dimensional
numerical vectors called embeddings. Unlike traditional relational databases that use exact
keyword matching, vector databases enable semantic similarity search — finding data based on
meaning rather than precise terms.

Each record stores three components: a unique ID, the numerical vector produced by an
embedding model, and optional metadata such as source filename or document category.

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
