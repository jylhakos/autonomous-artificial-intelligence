"""Create sample data for RAG pipeline testing.

Generates sample text files to demonstrate chunking and retrieval.
"""

import os


def create_sample_documents():
    """Create sample text documents for RAG testing."""
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Sample document 1: About RAG
    rag_content = """Retrieval-Augmented Generation (RAG) Overview

Retrieval-Augmented Generation is an architectural pattern that enhances Large Language Models
by combining them with external knowledge retrieval. This approach addresses several key limitations
of traditional LLMs.

Key Benefits:
1. Up-to-date Information: RAG systems can access current data without requiring model retraining.
2. Reduced Hallucinations: By grounding responses in actual documents, RAG systems produce more
   accurate and verifiable outputs.
3. Domain Specialization: Organizations can leverage their private knowledge bases and proprietary
   documentation.
4. Cost Efficiency: RAG eliminates the need for expensive model fine-tuning while still providing
   domain-specific capabilities.

How RAG Works:
The RAG process involves three main steps. First, relevant information is retrieved from a vector
database based on the user's query. Second, this retrieved context augments the LLM prompt. Third,
the model generates an accurate, grounded response using both the query and retrieved information.

Vector databases play a crucial role in RAG systems by enabling semantic search over large document
collections. Documents are converted into embeddings - high-dimensional vector representations that
capture semantic meaning. When a user submits a query, it's also converted to an embedding and
compared against stored document embeddings using similarity metrics like cosine similarity.
"""
    
    # Sample document 2: About Chunking
    chunking_content = """Document Chunking Strategies for RAG

Chunking is the process of breaking down large documents into smaller, manageable pieces called chunks.
This is a critical first step when preparing data for use with Large Language Models in RAG systems.

Why Chunking Matters:
Getting chunking right is one of the most important decisions in building your RAG pipeline. How you
split your documents directly affects your system's ability to find relevant information and generate
accurate answers.

Chunking impacts several key factors:
- Embedding Quality: Embedding models perform best on coherent, focused text segments
- Search Precision: Smaller chunks enable more precise semantic matching
- Context Window Limits: Chunks must fit within LLM token limits
- Cost Optimization: Smaller, targeted chunks reduce API token usage

Common Chunking Strategies:

1. Fixed-Size Chunking
Splits text by a set number of characters or tokens (e.g., 512 tokens per chunk). This is the
simplest approach and works well for uniformly structured documents. However, it may split
sentences or paragraphs mid-thought.

2. Sentence-Level Chunking
Breaks text into individual sentences or groups of sentences, preserving complete logical thoughts.
This works well for documents where sentence boundaries are important, like research papers.

3. Semantic Chunking
Uses AI to identify when the meaning or topic shifts, creating chunks based on natural theme breaks
rather than arbitrary character counts. This produces the highest quality semantic coherence but
requires more computational resources.

4. Document-Based Chunking
Respects the native structure of documents, using HTML tags, Markdown headers, or JSON keys to
define boundaries. Ideal for structured content like web pages or API documentation.

5. Recursive Chunking
Breaks text hierarchically using delimiters (paragraphs, then sentences, then words) until chunks
fit size constraints. This balances structure preservation with size requirements.

Best Practices:
- Use 10-20% overlap between chunks to preserve context at boundaries
- Include metadata like source, page number, and section headers
- Test different strategies with your specific content type
- Monitor retrieval quality metrics to optimize chunk size
"""
    
    # Sample document 3: About Evaluation
    evaluation_content = """RAG System Evaluation and Metrics

Evaluation is the systematic measurement of your RAG system's performance across retrieval quality,
context relevance, and answer accuracy. Without proper evaluation, you cannot optimize your system
for production use.

Why Evaluation Matters:
Evaluation enables you to:
- Determine if changes improve or degrade performance
- Compare different embedding models or chunking strategies
- Identify failure modes like irrelevant retrievals or hallucinations
- Optimize for both quality and cost

Key Evaluation Metrics:

Retrieval Metrics:
1. Precision@K: What proportion of the top-K retrieved chunks are actually relevant?
2. Recall@K: What proportion of all relevant chunks appear in the top-K results?
3. Mean Reciprocal Rank (MRR): How quickly does the first relevant chunk appear?
4. NDCG@K: Normalized measure accounting for graded relevance and ranking position

Generation Metrics:
1. Context Relevance: Are the retrieved chunks actually relevant to the query?
2. Answer Faithfulness: Is the answer grounded in the retrieved context without hallucinations?
3. Answer Relevance: Does the answer actually address the user's question?
4. Latency: How long does the end-to-end process take?

Evaluation Process:
Start by creating a test set of queries with known correct answers or relevant documents.
Run your RAG system on each query and measure both retrieval and generation quality.
Iterate on chunking strategies, retrieval parameters, and prompts based on metrics.

Continuous Monitoring:
In production, monitor these metrics continuously to detect degradation over time.
Track user feedback and implicit signals like engagement and satisfaction.
Use A/B testing to evaluate system changes before full deployment.
"""
    
    # Sample document 4: Vector Databases
    vectordb_content = """Vector Databases for RAG Systems

Vector databases are specialized storage systems optimized for storing and querying high-dimensional
vector embeddings. They form the foundation of Retrieval-Augmented Generation systems.

Popular Vector Databases:

1. Weaviate
Weaviate is an open-source vector database that supports automatic vectorization through modules
like text2vec-openai. It requires pre-chunked text but handles embedding generation automatically.
Weaviate emphasizes semantic chunking and late chunking strategies.

2. Pinecone
Pinecone is a managed vector database service. Unlike Weaviate, Pinecone does not split text
internally - you must chunk your text, embed each chunk individually, and store them as separate
records with IDs linking back to parent documents.

3. Chroma
Chroma is a lightweight, embedded vector database ideal for development and small-scale deployments.
It integrates seamlessly with LangChain and LlamaIndex.

4. Qdrant
Qdrant is a high-performance vector database with advanced filtering capabilities and hybrid search
support combining vector and keyword matching.

Integration with Frameworks:

LangChain Integration:
LangChain provides unified interfaces for multiple vector databases through its VectorStore abstraction.
You can switch between databases with minimal code changes.

LlamaIndex Integration:
To integrate LlamaIndex with a vector database, initialize the database's VectorStore class,
wrap it in a StorageContext, and pass it to VectorStoreIndex. LlamaIndex handles document parsing,
chunking via Node Parsers, and indexing automatically.

Key Considerations:
- Scalability: Can the database handle your expected document volume?
- Query Performance: What's the latency for similarity search?
- Filtering: Can you combine vector search with metadata filters?
- Cost: Managed services vs self-hosted deployment costs
- Integration: How well does it work with your chosen framework?
"""
    
    # Write files
    files = {
        "rag_overview.txt": rag_content,
        "chunking_strategies.txt": chunking_content,
        "evaluation_metrics.txt": evaluation_content,
        "vector_databases.txt": vectordb_content
    }
    
    for filename, content in files.items():
        filepath = os.path.join("data", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ Created {filepath}")
    
    print(f"\n✓ Successfully created {len(files)} sample documents in data/")
    print("\nYou can now run:")
    print("  python complete_rag_pipeline.py")


if __name__ == "__main__":
    create_sample_documents()
