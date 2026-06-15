"""RAG Evaluation Metrics and Methods

This module provides evaluation metrics for Retrieval-Augmented Generation systems,
including retrieval accuracy, context relevance, and answer quality metrics.

Key Metrics:
1. Retrieval Accuracy: Measures if the system finds the right information
2. Context Relevance: Evaluates quality of retrieved chunks
3. Answer Faithfulness: Checks if answers are grounded in retrieved context
4. Answer Relevance: Measures if answers address the user's question

References:
- https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-preparation-phase
- https://www.pinecone.io/learn/chunking-strategies/
"""

from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float


@dataclass
class RAGMetrics:
    """Comprehensive metrics for RAG system evaluation."""
    retrieval_accuracy: float
    context_relevance: float
    answer_faithfulness: float
    answer_relevance: float
    latency_ms: float


def calculate_precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    """Calculate Precision@K for retrieval.
    
    Precision@K measures what proportion of the top K retrieved documents are relevant.
    
    Args:
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        relevant_docs: List of known relevant document IDs
        k: Number of top results to consider
    
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k == 0 or len(retrieved_docs) == 0:
        return 0.0
    
    top_k = retrieved_docs[:k]
    relevant_in_top_k = len([doc for doc in top_k if doc in relevant_docs])
    return relevant_in_top_k / k


def calculate_recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    """Calculate Recall@K for retrieval.
    
    Recall@K measures what proportion of all relevant documents appear in the top K results.
    
    Args:
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        relevant_docs: List of known relevant document IDs
        k: Number of top results to consider
    
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if len(relevant_docs) == 0:
        return 0.0
    
    top_k = retrieved_docs[:k]
    relevant_in_top_k = len([doc for doc in top_k if doc in relevant_docs])
    return relevant_in_top_k / len(relevant_docs)


def calculate_mrr(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).
    
    MRR measures how quickly the system returns a relevant result.
    Score is 1/rank of first relevant document.
    
    Args:
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        relevant_docs: List of known relevant document IDs
    
    Returns:
        MRR score (0.0 to 1.0)
    """
    for i, doc in enumerate(retrieved_docs, 1):
        if doc in relevant_docs:
            return 1.0 / i
    return 0.0


def calculate_ndcg_at_k(retrieved_docs: List[str], relevance_scores: Dict[str, float], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K).
    
    NDCG measures ranking quality with graded relevance scores.
    Accounts for position bias (higher-ranked results matter more).
    
    Args:
        retrieved_docs: List of retrieved document IDs (ordered by relevance)
        relevance_scores: Dictionary mapping doc IDs to relevance scores (0-1 or 0-3)
        k: Number of top results to consider
    
    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    def dcg_at_k(docs, scores, k):
        dcg = 0.0
        for i, doc in enumerate(docs[:k], 1):
            relevance = scores.get(doc, 0.0)
            dcg += relevance / np.log2(i + 1)
        return dcg
    
    # Calculate DCG for retrieved docs
    dcg = dcg_at_k(retrieved_docs, relevance_scores, k)
    
    # Calculate ideal DCG (perfect ranking)
    ideal_docs = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
    idcg = dcg_at_k(ideal_docs, relevance_scores, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(retrieved_docs: List[str], 
                      relevant_docs: List[str],
                      relevance_scores: Dict[str, float],
                      k: int = 5) -> RetrievalMetrics:
    """Comprehensive retrieval evaluation.
    
    Optimizing for Retrieval Accuracy:
    The first step is making sure your system can find the right information 
    in your vector database. Vector search does this by comparing user queries 
    with the embeddings of your chunks.
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of known relevant document IDs
        relevance_scores: Dictionary of graded relevance scores
        k: Number of top results to evaluate
    
    Returns:
        RetrievalMetrics with all computed scores
    """
    return RetrievalMetrics(
        precision_at_k=calculate_precision_at_k(retrieved_docs, relevant_docs, k),
        recall_at_k=calculate_recall_at_k(retrieved_docs, relevant_docs, k),
        mean_reciprocal_rank=calculate_mrr(retrieved_docs, relevant_docs),
        ndcg_at_k=calculate_ndcg_at_k(retrieved_docs, relevance_scores, k)
    )


def evaluate_context_relevance(retrieved_chunks: List[str], query: str, llm_evaluator) -> float:
    """Evaluate if retrieved chunks are relevant to the query.
    
    Uses an LLM to judge relevance of each retrieved chunk.
    
    Args:
        retrieved_chunks: List of text chunks retrieved from vector DB
        query: User's query
        llm_evaluator: LLM instance for evaluation
    
    Returns:
        Average relevance score (0.0 to 1.0)
    """
    relevance_scores = []
    
    for chunk in retrieved_chunks:
        prompt = f"""Rate the relevance of this text chunk to the query on a scale of 0-1.
        
Query: {query}
        
Chunk: {chunk}
        
Provide only a number between 0 and 1."""
        
        # Get LLM evaluation (simplified - real implementation would call LLM)
        # score = llm_evaluator.evaluate(prompt)
        # relevance_scores.append(score)
        pass
    
    return np.mean(relevance_scores) if relevance_scores else 0.0


def evaluate_answer_faithfulness(answer: str, context_chunks: List[str], llm_evaluator) -> float:
    """Evaluate if the generated answer is grounded in the retrieved context.
    
    Checks for hallucination - ensures answer doesn't contain information 
    not present in the context.
    
    Args:
        answer: Generated answer from RAG system
        context_chunks: Retrieved context chunks used to generate answer
        llm_evaluator: LLM instance for evaluation
    
    Returns:
        Faithfulness score (0.0 to 1.0)
    """
    context = "\n".join(context_chunks)
    
    prompt = f"""Evaluate if the answer is faithful to the context (no hallucinations).
    Rate 0 (completely unfaithful) to 1 (completely faithful).
    
    Context: {context}
    
    Answer: {answer}
    
    Provide only a number between 0 and 1."""
    
    # Get LLM evaluation (simplified)
    # faithfulness = llm_evaluator.evaluate(prompt)
    # return faithfulness
    return 0.0


def evaluate_answer_relevance(answer: str, query: str, llm_evaluator) -> float:
    """Evaluate if the answer actually addresses the user's question.
    
    Args:
        answer: Generated answer from RAG system
        query: Original user query
        llm_evaluator: LLM instance for evaluation
    
    Returns:
        Relevance score (0.0 to 1.0)
    """
    prompt = f"""Rate how well this answer addresses the question on a scale of 0-1.
    
    Question: {query}
    
    Answer: {answer}
    
    Provide only a number between 0 and 1."""
    
    # Get LLM evaluation (simplified)
    # relevance = llm_evaluator.evaluate(prompt)
    # return relevance
    return 0.0


if __name__ == "__main__":
    # Example usage
    print("RAG Evaluation Example\n")
    
    # Sample data
    retrieved = ["doc1", "doc3", "doc5", "doc2", "doc7"]
    relevant = ["doc1", "doc2", "doc4"]
    relevance_scores = {
        "doc1": 1.0,
        "doc2": 0.8,
        "doc3": 0.0,
        "doc4": 0.9,
        "doc5": 0.2,
        "doc7": 0.0
    }
    
    # Evaluate retrieval
    metrics = evaluate_retrieval(retrieved, relevant, relevance_scores, k=5)
    
    print("Retrieval Metrics:")
    print(f"  Precision@5: {metrics.precision_at_k:.3f}")
    print(f"  Recall@5: {metrics.recall_at_k:.3f}")
    print(f"  MRR: {metrics.mean_reciprocal_rank:.3f}")
    print(f"  NDCG@5: {metrics.ndcg_at_k:.3f}")
