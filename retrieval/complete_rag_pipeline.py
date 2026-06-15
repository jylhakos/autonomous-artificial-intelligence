"""Complete End-to-End RAG Pipeline Example

This module demonstrates a complete RAG (Retrieval-Augmented Generation) pipeline
from data preparation through evaluation, integrating:
- Document loading and chunking
- Vector database indexing (Weaviate)
- Retrieval and generation
- Comprehensive evaluation

References:
- https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase
- https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-preparation-phase
- https://weaviate.io/blog/chunking-strategies-for-rag
- https://www.pinecone.io/learn/chunking-strategies/
"""

import os
from typing import List, Dict
import weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dataclasses import dataclass
import time


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 3
    collection_name: str = "RAGDocuments"
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-4"


class RAGPipeline:
    """Complete RAG pipeline implementation."""
    
    def __init__(self, config: RAGConfig):
        """Initialize RAG pipeline with configuration.
        
        Args:
            config: RAG configuration parameters
        """
        self.config = config
        self.weaviate_client = None
        self.embeddings_model = None
        self.llm = None
        self.chunks = []
        
    def initialize_models(self):
        """Initialize embedding and LLM models."""
        print("Initializing models...")
        
        # Initialize OpenAI embeddings
        self.embeddings_model = OpenAIEmbeddings(
            model=self.config.embedding_model
        )
        
        # Initialize LLM for generation
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=0.0
        )
        
        print("✓ Models initialized")
    
    def connect_vector_db(self):
        """Connect to Weaviate vector database."""
        print("Connecting to Weaviate...")
        
        try:
            self.weaviate_client = weaviate.connect_to_local()
            print("✓ Connected to Weaviate")
        except Exception as e:
            print(f"✗ Failed to connect to Weaviate: {e}")
            raise
    
    def load_documents(self, data_path: str) -> List:
        """Load documents from directory.
        
        Args:
            data_path: Path to directory containing documents
        
        Returns:
            List of loaded documents
        """
        print(f"Loading documents from {data_path}...")
        
        try:
            loader = DirectoryLoader(
                data_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
            print(f"✓ Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            print(f"✗ Failed to load documents: {e}")
            raise
    
    def chunk_documents(self, documents: List) -> List:
        """Apply chunking strategy to documents.
        
        This implements Recursive Chunking, which breaks text down
        hierarchically using a list of delimiters until chunks fit
        the size limit.
        
        Reference: https://weaviate.io/blog/chunking-strategies-for-rag
        
        Args:
            documents: List of documents to chunk
        
        Returns:
            List of document chunks
        """
        print(f"Chunking documents (size={self.config.chunk_size}, overlap={self.config.chunk_overlap})...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\\n\\n", "\\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        self.chunks = chunks
        
        print(f"✓ Created {len(chunks)} chunks")
        return chunks
    
    def index_chunks(self, chunks: List):
        """Index chunks into Weaviate vector database.
        
        Weaviate requires pre-chunked text and will auto-vectorize
        via configured modules like text2vec-openai.
        
        Reference: https://weaviate.io/blog/chunking-strategies-for-rag
        
        Args:
            chunks: List of document chunks to index
        """
        print(f"Indexing {len(chunks)} chunks into Weaviate...")
        
        try:
            # Get or create collection
            collection = self.weaviate_client.collections.get(
                self.config.collection_name
            )
            
            # Batch insert chunks
            with collection.batch.dynamic() as batch:
                for i, chunk in enumerate(chunks):
                    # Generate embedding
                    embedding = self.embeddings_model.embed_query(
                        chunk.page_content
                    )
                    
                    # Add to batch
                    batch.add_object(
                        properties={
                            "content": chunk.page_content,
                            "source": chunk.metadata.get("source", "unknown"),
                            "chunk_id": i
                        },
                        vector=embedding
                    )
            
            print(f"✓ Indexed {len(chunks)} chunks")
            
        except Exception as e:
            print(f"✗ Failed to index chunks: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve relevant chunks for a query.
        
        Optimizing for Retrieval Accuracy:
        Vector search compares user queries with the embeddings of chunks
        to find the most relevant information.
        
        Reference: 
        - https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-preparation-phase
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve (default: config value)
        
        Returns:
            List of retrieved chunks with metadata
        """
        if top_k is None:
            top_k = self.config.top_k_retrieval
        
        print(f"Retrieving top-{top_k} chunks for query: '{query}'")
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.embed_query(query)
            
            # Search vector database
            collection = self.weaviate_client.collections.get(
                self.config.collection_name
            )
            
            results = collection.query.near_vector(
                near_vector=query_embedding,
                limit=top_k
            )
            
            retrieved_chunks = [
                {
                    "content": obj.properties["content"],
                    "source": obj.properties["source"],
                    "chunk_id": obj.properties["chunk_id"]
                }
                for obj in results.objects
            ]
            
            print(f"✓ Retrieved {len(retrieved_chunks)} chunks")
            return retrieved_chunks
            
        except Exception as e:
            print(f"✗ Failed to retrieve chunks: {e}")
            raise
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using retrieved context.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
        
        Returns:
            Generated answer string
        """
        print("Generating answer...")
        
        # Combine context chunks
        context = "\\n\\n".join([
            f"[Source: {chunk['source']}]\\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        # Create prompt
        prompt = f"""Use the following context to answer the question. 
If you cannot answer based on the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        response = self.llm.invoke(prompt)
        answer = response.content
        
        print("✓ Answer generated")
        return answer
    
    def query(self, query: str) -> Dict:
        """Complete RAG query: retrieve + generate.
        
        Args:
            query: User query string
        
        Returns:
            Dictionary with answer, context, and metadata
        """
        start_time = time.time()
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(query)
        
        # Generate answer
        answer = self.generate_answer(query, retrieved_chunks)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "query": query,
            "answer": answer,
            "context_chunks": retrieved_chunks,
            "latency_ms": latency_ms,
            "num_chunks_retrieved": len(retrieved_chunks)
        }
    
    def evaluate(self, test_queries: List[Dict]) -> Dict:
        """Evaluate RAG pipeline on test queries.
        
        What is Evaluation?
        Evaluation is the systematic measurement of your RAG system's
        performance across retrieval quality, context relevance, and
        answer accuracy.
        
        Args:
            test_queries: List of test queries with expected answers
        
        Returns:
            Evaluation metrics
        """
        print(f"\\nEvaluating pipeline on {len(test_queries)} queries...")
        
        total_latency = 0
        results = []
        
        for test_case in test_queries:
            query = test_case["query"]
            result = self.query(query)
            
            total_latency += result["latency_ms"]
            results.append({
                "query": query,
                "answer": result["answer"],
                "latency_ms": result["latency_ms"]
            })
        
        avg_latency = total_latency / len(test_queries)
        
        metrics = {
            "num_queries": len(test_queries),
            "avg_latency_ms": avg_latency,
            "results": results
        }
        
        print(f"✓ Evaluation complete")
        print(f"  Average latency: {avg_latency:.2f}ms")
        
        return metrics
    
    def close(self):
        """Clean up resources."""
        if self.weaviate_client:
            self.weaviate_client.close()
            print("✓ Closed Weaviate connection")


def main():
    """Example usage of complete RAG pipeline."""
    
    print("=" * 60)
    print("Complete RAG Pipeline Example")
    print("=" * 60)
    print()
    
    # Configure pipeline
    config = RAGConfig(
        chunk_size=512,
        chunk_overlap=50,
        top_k_retrieval=3,
        collection_name="RAGDemo"
    )
    
    # Initialize pipeline
    pipeline = RAGPipeline(config)
    
    try:
        # Step 1: Initialize models
        pipeline.initialize_models()
        print()
        
        # Step 2: Connect to vector database
        pipeline.connect_vector_db()
        print()
        
        # Step 3: Load documents
        # Note: Create a 'data/' directory with sample .txt files
        documents = pipeline.load_documents("data/")
        print()
        
        # Step 4: Chunk documents
        # Why is Chunking important for RAG?
        # Getting chunking right is one of the most important decisions
        # in building your RAG pipeline. How you split your documents
        # affects your system's ability to find relevant information
        # and give accurate answers.
        chunks = pipeline.chunk_documents(documents)
        print()
        
        # Step 5: Index chunks
        pipeline.index_chunks(chunks)
        print()
        
        # Step 6: Query the system
        print("=" * 60)
        print("Querying RAG System")
        print("=" * 60)
        print()
        
        test_query = "What are the key concepts in these documents?"
        result = pipeline.query(test_query)
        
        print(f"\\nQuery: {result['query']}")
        print(f"\\nAnswer: {result['answer']}")
        print(f"\\nLatency: {result['latency_ms']:.2f}ms")
        print(f"Chunks retrieved: {result['num_chunks_retrieved']}")
        print()
        
        # Step 7: Evaluation
        print("=" * 60)
        print("Evaluation")
        print("=" * 60)
        
        test_queries = [
            {"query": "What are the main topics discussed?"},
            {"query": "Can you summarize the key points?"},
            {"query": "What recommendations are provided?"}
        ]
        
        metrics = pipeline.evaluate(test_queries)
        
        print(f"\\nEvaluation Results:")
        print(f"  Total queries: {metrics['num_queries']}")
        print(f"  Average latency: {metrics['avg_latency_ms']:.2f}ms")
        
    except Exception as e:
        print(f"\\n✗ Pipeline failed: {e}")
        
    finally:
        # Cleanup
        pipeline.close()
        print()
        print("=" * 60)
        print("Pipeline Complete")
        print("=" * 60)


if __name__ == "__main__":
    main()
