"""Chunking Implementation with LangChain for RAG Systems

This module demonstrates different chunking strategies for Retrieval-Augmented Generation (RAG)
using LangChain, compatible with both Pinecone and Weaviate vector databases.

References:
- https://www.pinecone.io/learn/chunking-strategies/
- https://weaviate.io/blog/chunking-strategies-for-rag
- https://docs.pinecone.io/guides/index-data/data-modeling
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from typing import List
import weaviate
import os


def fixed_size_chunking(docs: List, chunk_size: int = 512, chunk_overlap: int = 50):
    """Fixed-Size Chunking: Splits text by a set number of characters or tokens.
    
    Best for: Basic documents with uniform structure.
    
    Args:
        docs: List of documents to chunk
        chunk_size: Target max characters per chunk (e.g., 512 tokens)
        chunk_overlap: Overlap between chunks to retain context (e.g., 50-100 tokens)
    
    Returns:
        List of chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


def recursive_chunking(docs: List, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Recursive Chunking: Breaks text hierarchically using delimiters.
    
    Best for: Varied, unstructured text with natural paragraph breaks.
    Uses separators like paragraphs, then sentences until chunk fits size limit.
    
    Args:
        docs: List of documents to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters of overlap between chunks
    
    Returns:
        List of chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


def ingest_to_weaviate(chunks: List, collection_name: str = "Question"):
    """Ingest chunks into Weaviate vector database.
    
    Weaviate's Python client auto-vectorizes text via configured modules 
    (like text2vec-openai), but requires pre-chunked text.
    
    Reference: https://weaviate.io/blog/chunking-strategies-for-rag
    
    Args:
        chunks: List of document chunks
        collection_name: Name of Weaviate collection
    """
    # Initialize Weaviate client
    client = weaviate.connect_to_local()
    
    try:
        questions = client.collections.get(collection_name)
        
        # Iterate through chunks and insert into Weaviate
        with questions.batch.dynamic() as batch:
            for i, chunk in enumerate(chunks):
                batch.add_object(
                    properties={
                        "content": chunk.page_content,
                        "source": chunk.metadata.get("source", "unknown"),
                        "chunk_id": i
                    }
                )
        print(f"Successfully ingested {len(chunks)} chunks into Weaviate")
    finally:
        client.close()


def ingest_to_pinecone(chunks: List, embeddings_model, index_name: str):
    """Ingest chunks into Pinecone vector database.
    
    Pinecone does not split text internally; you must chunk your text, 
    embed each chunk individually, and store them as separate records 
    with IDs that connect back to the parent document.
    
    Reference: https://docs.pinecone.io/guides/index-data/data-modeling
    
    Args:
        chunks: List of document chunks
        embeddings_model: OpenAI or other embeddings model
        index_name: Name of Pinecone index
    """
    import pinecone
    
    # Initialize Pinecone
    pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    
    # Prepare vectors for upsert
    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        # Generate embedding for chunk
        embedding = embeddings_model.embed_query(chunk.page_content)
        
        # Create vector record
        vectors_to_upsert.append({
            "id": f"doc_{chunk.metadata.get('source', 'unknown')}_chunk_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "chunk_index": i
            }
        })
    
    # Upsert to Pinecone
    index.upsert(vectors=vectors_to_upsert)
    print(f"Successfully ingested {len(chunks)} chunks into Pinecone")


if __name__ == "__main__":
    # Example usage
    
    # 1. Load your document
    loader = TextLoader("your_document.txt")
    docs = loader.load()
    
    # 2. Choose chunking strategy
    print("Using Fixed-Size Chunking...")
    chunks_fixed = fixed_size_chunking(docs, chunk_size=500, chunk_overlap=50)
    print(f"Created {len(chunks_fixed)} fixed-size chunks")
    
    print("\nUsing Recursive Chunking...")
    chunks_recursive = recursive_chunking(docs, chunk_size=1000, chunk_overlap=200)
    print(f"Created {len(chunks_recursive)} recursive chunks")
    
    # 3. Initialize embedding model
    embeddings_model = OpenAIEmbeddings()
    
    # 4. Ingest into vector database (choose one)
    # ingest_to_weaviate(chunks_recursive, collection_name="Documents")
    # ingest_to_pinecone(chunks_recursive, embeddings_model, index_name="rag-index")
