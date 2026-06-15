"""Chunking Implementation with LlamaIndex for RAG Systems

Demonstrates how to use LlamaIndex with Azure AI Foundry (for models and embeddings)
and Weaviate (as the vector database) for Retrieval-Augmented Generation.

References:
- https://learn.microsoft.com/en-us/azure/storage/files/artificial-intelligence/retrieval-augmented-generation/open-source-frameworks/tutorials/llamaindex-weaviate/tutorial-llamaindex-weaviate
- https://learn.microsoft.com/en-us/azure/foundry-classic/how-to/develop/llama-index
- https://developers.llamaindex.ai/python/framework/integrations/vector_stores/weaviateindex_auto_retriever/
- https://developers.llamaindex.ai/python/framework/optimizing/basic_strategies/basic_strategies/
- https://developers.llamaindex.ai/python/framework/community/integrations/vector_stores/
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
import weaviate
import os


def setup_azure_models():
    """Configure Azure OpenAI models for embeddings and LLM.
    
    Returns:
        Tuple of (embeddings_model, llm)
    """
    # Configure Azure OpenAI Embeddings
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name=os.environ.get("AZURE_EMBEDDING_DEPLOYMENT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-02-15-preview",
    )
    
    # Configure Azure OpenAI LLM
    llm = AzureOpenAI(
        model="gpt-4",
        deployment_name=os.environ.get("AZURE_LLM_DEPLOYMENT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-02-15-preview",
    )
    
    return embed_model, llm


def sentence_chunking(documents, chunk_size: int = 512, chunk_overlap: int = 50):
    """Sentence-Level Chunking with LlamaIndex.
    
    Breaks text into individual sentences or groups of sentences,
    keeping complete logical thoughts together.
    
    Args:
        documents: List of documents from SimpleDirectoryReader
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
    
    Returns:
        List of nodes (chunks)
    """
    # Initialize SentenceSplitter
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    # Create nodes from documents
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes


def semantic_chunking(documents, embed_model, buffer_size: int = 1):
    """Semantic Chunking with LlamaIndex.
    
    Uses AI to identify when the actual meaning or topic of the text shifts,
    creating chunks based on natural paragraph or theme breaks rather than
    arbitrary character counts.
    
    Reference: https://weaviate.io/blog/chunking-strategies-for-rag
    
    Args:
        documents: List of documents from SimpleDirectoryReader
        embed_model: Embedding model for semantic analysis
        buffer_size: Number of sentences to group for comparison
    
    Returns:
        List of semantically-chunked nodes
    """
    # Initialize Semantic Splitter
    splitter = SemanticSplitterNodeParser(
        buffer_size=buffer_size,
        embed_model=embed_model,
    )
    
    # Create semantically-aware nodes
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes


def create_weaviate_index(nodes, embed_model, collection_name: str = "RAGDocuments"):
    """Create a LlamaIndex VectorStoreIndex with Weaviate backend.
    
    To integrate LlamaIndex with a vector database, you initialize the
    database's specific VectorStore class, wrap it inside a LlamaIndex
    StorageContext, and pass that context into a VectorStoreIndex.
    
    Reference:
    - https://developers.llamaindex.ai/python/framework/community/integrations/vector_stores/
    
    Args:
        nodes: List of document nodes/chunks
        embed_model: Azure OpenAI embedding model
        collection_name: Name of Weaviate collection
    
    Returns:
        VectorStoreIndex instance
    """
    # Initialize Weaviate client
    client = weaviate.connect_to_local()
    
    # Create Weaviate vector store
    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name=collection_name,
    )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create and populate index
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    
    return index


def build_query_engine(index, llm, similarity_top_k: int = 3):
    """Build a retrieval query engine from the index.
    
    LlamaIndex provides tools to define an advanced retrieval/query engine
    over your data. The retriever constructs retrieve data from your
    knowledge base given an input prompt.
    
    Reference:
    - https://developers.llamaindex.ai/python/framework/integrations/vector_stores/weaviateindex_auto_retriever/
    
    Args:
        index: VectorStoreIndex instance
        llm: Language model for response generation
        similarity_top_k: Number of similar chunks to retrieve
    
    Returns:
        Query engine instance
    """
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=similarity_top_k,
    )
    return query_engine


if __name__ == "__main__":
    # Example: Build a RAG pipeline with LlamaIndex and Weaviate
    
    # 1. Load documents
    print("Loading documents...")
    documents = SimpleDirectoryReader("data").load_data()
    
    # 2. Setup Azure models
    print("Setting up Azure OpenAI models...")
    embed_model, llm = setup_azure_models()
    
    # 3. Choose chunking strategy
    print("\nApplying Sentence Chunking...")
    nodes_sentence = sentence_chunking(documents, chunk_size=512, chunk_overlap=50)
    print(f"Created {len(nodes_sentence)} sentence-based chunks")
    
    print("\nApplying Semantic Chunking...")
    nodes_semantic = semantic_chunking(documents, embed_model, buffer_size=1)
    print(f"Created {len(nodes_semantic)} semantic chunks")
    
    # 4. Create Weaviate index
    print("\nCreating Weaviate index...")
    index = create_weaviate_index(nodes_semantic, embed_model, "RAGDemo")
    
    # 5. Build query engine
    print("Building query engine...")
    query_engine = build_query_engine(index, llm, similarity_top_k=3)
    
    # 6. Query the system
    print("\nQuerying RAG system...")
    response = query_engine.query("What are the key concepts in these documents?")
    print(f"Response: {response}")
