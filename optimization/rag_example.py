"""
rag_example.py
==============
Demonstrates a minimal Retrieval-Augmented Generation (RAG) pipeline using
local sentence embeddings (all-MiniLM-L6-v2) and FAISS for vector similarity
search.

README section: "Example Usage: RAG for Accurate Knowledge Retrieval"

Concept covered:
    RAG inserts relevant context retrieved from an external knowledge base into
    the prompt before sending it to the LLM.  This reduces hallucinations and
    keeps answers grounded in factual, up-to-date information without requiring
    fine-tuning.

    This script covers the local FAISS-based variant.  The README also describes
    a production-grade equivalent using Azure AI Search
    (section "RAG Optimization with Azure AI Search") where the same
    architecture is scaled to millions of documents with semantic ranking.

    Steps implemented:
        1. Encode a small knowledge base into dense vector embeddings.
        2. Build a FAISS IndexFlatL2 for fast L2 nearest-neighbour search.
        3. Encode a user query and retrieve the top-k most relevant passages.
        4. Construct an augmented prompt containing the retrieved context.

Requirements:
    - Virtual environment activated (see README "Python Virtual Environment Setup")

    pip install sentence-transformers faiss-cpu numpy
    # or:
    pip install -r requirements.txt

Usage:
    # 1. Activate the virtual environment
    source .venv/bin/activate

    # 2. Run the script
    python rag_example.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ---------------------------------------------------------------------------
# Step 1: Define a small knowledge base
# In production this would be a large corpus (e.g., indexed with Azure AI Search)
# ---------------------------------------------------------------------------
documents = [
    "Quantization reduces model precision from FP32 to INT8 or INT4.",
    "LoRA fine-tunes models by injecting small trainable adapter matrices.",
    "Pruning removes low-importance weights from a neural network.",
    "Knowledge distillation trains a small student model from a large teacher model.",
    "PagedAttention manages KV cache memory for efficient LLM serving.",
]

# ---------------------------------------------------------------------------
# Step 2: Encode documents into dense vector embeddings
# ---------------------------------------------------------------------------
print("Loading embedding model (all-MiniLM-L6-v2) ...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedding_model.encode(documents, convert_to_numpy=True)
print(f"Encoded {len(documents)} documents, embedding dimension: {doc_embeddings.shape[1]}")

# ---------------------------------------------------------------------------
# Step 3: Build a FAISS index for fast similarity search
# IndexFlatL2 performs exact L2 (Euclidean) nearest-neighbour search.
# For millions of documents, use IndexIVFFlat or IndexHNSWFlat instead.
# ---------------------------------------------------------------------------
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)
print(f"FAISS index built with {index.ntotal} vectors")


def retrieve(query: str, top_k: int = 2) -> list[str]:
    """Retrieve the top_k most relevant documents for a given query.

    Args:
        query:  Natural language question or search phrase.
        top_k:  Number of documents to return.

    Returns:
        List of the top_k document strings ranked by similarity.
    """
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]


# ---------------------------------------------------------------------------
# Step 4: Retrieve relevant context and build the augmented prompt
# ---------------------------------------------------------------------------
user_question = "How does LoRA reduce memory usage during fine-tuning?"
retrieved_docs = retrieve(user_question, top_k=2)

print(f"\nUser question: {user_question}")
print("\nRetrieved documents:")
for doc in retrieved_docs:
    print(f"  - {doc}")

context = "\n".join(f"- {doc}" for doc in retrieved_docs)
augmented_prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {user_question}
Answer:"""

print("\nAugmented prompt sent to the LLM:")
print(augmented_prompt)
# In a full implementation, pass augmented_prompt to your LLM for generation.
# Example:
#   from transformers import pipeline
#   generator = pipeline("text-generation", model="gpt2")
#   response = generator(augmented_prompt, max_new_tokens=150)
#   print(response[0]["generated_text"])
