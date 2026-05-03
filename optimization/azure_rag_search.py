"""
azure_rag_search.py
===================
Demonstrates a production RAG pipeline using Azure AI Search for multi-vector
retrieval and Azure OpenAI Service for grounded answer generation.

README section: "RAG Optimization with Azure AI Search"

Concept covered:
    This script is the production-scale counterpart to rag_example.py, which
    uses a local FAISS index.  Azure AI Search provides:
    - Managed vector indexing at scale (millions of documents)
    - Semantic ranking of retrieved results
    - Hybrid search (keyword + vector)
    - No infrastructure management required

    The overall RAG architecture is:
        User Query
            → Azure AI Search (multi-vector + semantic ranking)
            → Top-k relevant document passages
            → Augmented prompt (context + question)
            → Azure OpenAI GPT-4o
            → Grounded response

Related README sections:
    - "Example Usage: RAG for Accurate Knowledge Retrieval" — local FAISS variant
    - "RAG Optimization with Azure AI Search"              — architecture diagram
    - "Optimization on Microsoft Azure (Cloud)"            — Azure ecosystem

Requirements:
    - Azure AI Search resource with a populated index
    - Azure OpenAI Service resource with a GPT-4o deployment
    - Virtual environment activated (see README "Python Virtual Environment Setup")

    pip install azure-search-documents azure-identity openai
    # or:
    pip install -r requirements.txt

Usage:
    # 1. Activate the virtual environment
    source .venv/bin/activate

    # 2. Set environment variables (do NOT hard-code credentials in source files)
    export AZURE_SEARCH_ENDPOINT="https://<search-service>.search.windows.net"
    export AZURE_SEARCH_KEY="<your-search-api-key>"
    export AZURE_SEARCH_INDEX="<your-index-name>"
    export AZURE_OPENAI_ENDPOINT="https://<endpoint>.openai.azure.com/"
    export AZURE_OPENAI_KEY="<your-openai-api-key>"

    # 3. Run the script
    python azure_rag_search.py
"""

import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Read credentials from environment variables
# ---------------------------------------------------------------------------
search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
search_key      = os.environ["AZURE_SEARCH_KEY"]
index_name      = os.environ["AZURE_SEARCH_INDEX"]
openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
openai_key      = os.environ["AZURE_OPENAI_KEY"]

# ---------------------------------------------------------------------------
# Initialise clients
# ---------------------------------------------------------------------------
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_key),
)

openai_client = AzureOpenAI(
    api_key=openai_key,
    api_version="2024-02-01",
    azure_endpoint=openai_endpoint,
)


def rag_query(user_question: str, top_k: int = 3) -> str:
    """Run a RAG query against Azure AI Search + Azure OpenAI.

    Args:
        user_question: The question to answer.
        top_k:         Number of documents to retrieve from the search index.

    Returns:
        The model's grounded answer as a string.
    """
    # Step 1: Retrieve relevant context from Azure AI Search
    results = search_client.search(
        search_text=user_question,
        top=top_k,
        query_type="semantic",
        semantic_configuration_name="default",
    )
    context = "\n".join(r["content"] for r in results)

    # Step 2: Build augmented prompt and send to Azure OpenAI
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Answer questions based on the provided context only. "
                           "If the context does not contain the answer, say so.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {user_question}",
            },
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
question = "What optimization techniques reduce VRAM usage during inference?"
print(f"Question: {question}\n")
answer = rag_query(question)
print(f"Answer:\n{answer}")
