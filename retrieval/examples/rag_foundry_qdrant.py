#!/usr/bin/env python3
"""
Local RAG example — Qdrant (Docker) + Microsoft Foundry Local
=============================================================

This script demonstrates a fully offline Retrieval-Augmented Generation (RAG)
pipeline using:

  - Qdrant       : open-source vector database running in a local Docker container
  - Foundry Local: Microsoft's on-device AI runtime (OpenAI-compatible REST API)
  - all-MiniLM-L6-v2: lightweight local embedding model from sentence-transformers

Pipeline
--------
  1. Load .txt / .md documents from the docs/ folder next to this script.
  2. Split each document into overlapping text chunks.
  3. Embed every chunk using the local sentence-transformers model.
  4. Upsert the vectors and metadata into a Qdrant collection.
  5. Accept questions interactively.
  6. For each question: embed it, retrieve the top-k similar chunks from Qdrant,
     inject them as context into the Foundry Local model, and stream the answer.

Requirements
------------
  pip install -r requirements.txt   (from the project root)

Prerequisites
-------------
  Start Qdrant:       docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
  Start Foundry:      foundry service start
  Load a model:       foundry model run phi-3.5-mini

Usage
-----
  source venv/bin/activate          # activate the virtual environment first
  python examples/rag_foundry_qdrant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration — adjust these constants to match your local setup
# ---------------------------------------------------------------------------

# Microsoft Foundry Local exposes an OpenAI-compatible REST API at this URL.
# Run `foundry service start` to start the service.
FOUNDRY_BASE_URL: str = "http://localhost:5272/v1"

# Foundry Local accepts any non-empty string as the API key.
FOUNDRY_API_KEY: str = "foundry-local"

# The model alias used when running `foundry model run <alias>`.
# Change this if you loaded a different model, e.g. "phi-4-mini".
FOUNDRY_MODEL: str = "phi-3.5-mini"

# Qdrant Docker container is accessible at this URL after:
#   docker run -d -p 6333:6333 qdrant/qdrant
QDRANT_URL: str = "http://localhost:6333"

# Name of the Qdrant collection that stores the document embeddings.
COLLECTION_NAME: str = "rag_documents"

# Sentence-transformers model used for embedding.  The first run will
# download the model (~90 MB) and cache it in ~/.cache/torch/sentence_transformers/.
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Embedding dimension produced by all-MiniLM-L6-v2.
EMBEDDING_DIM: int = 384

# Maximum number of characters per text chunk.
CHUNK_SIZE: int = 400

# Number of overlapping characters between consecutive chunks.
# Overlap prevents relevant content from being split across chunk boundaries.
CHUNK_OVERLAP: int = 50

# Number of chunks to retrieve per question (top-k nearest neighbours).
TOP_K: int = 5

# Folder containing the documents to ingest.
# Place your .txt or .md files here before running the script.
DOCS_FOLDER: Path = Path(__file__).parent / "docs"


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    """Split *text* into overlapping fixed-size character-level chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------

def ingest_documents(
    qdrant: QdrantClient,
    embedder: SentenceTransformer,
) -> int:
    """Load documents from DOCS_FOLDER and upsert their chunks into Qdrant.

    Returns the total number of chunks stored.  Running the script a second
    time with the same documents is safe: upsert with the same integer IDs
    replaces the existing points rather than creating duplicates.
    """
    if not DOCS_FOLDER.exists():
        print(f"Docs folder not found: {DOCS_FOLDER}")
        print("Create the folder and add .txt or .md files to it.")
        return 0

    files = sorted(DOCS_FOLDER.glob("**/*.txt")) + sorted(
        DOCS_FOLDER.glob("**/*.md")
    )
    if not files:
        print(f"No .txt or .md files found in {DOCS_FOLDER}")
        return 0

    # Create the Qdrant collection if it does not already exist.
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION_NAME}'.")

    points: list[PointStruct] = []
    point_id = 1

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            vector = embedder.encode(chunk).tolist()
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "filename": file_path.name,
                        "chunk_index": idx,
                        "text": chunk,
                    },
                )
            )
            point_id += 1
        print(f"  Processed '{file_path.name}': {len(chunks)} chunk(s).")

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(
    qdrant: QdrantClient,
    embedder: SentenceTransformer,
    question: str,
) -> str:
    """Embed *question* and return the top-k most similar text chunks from Qdrant."""
    query_vector = embedder.encode(question).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    )
    chunks = [r.payload["text"] for r in results.points if r.payload]
    return "\n\n---\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(llm: OpenAI, context: str, question: str) -> None:
    """Send the retrieved *context* and *question* to Foundry Local and stream
    the model's response to stdout."""
    system_content = (
        "You are a helpful assistant. Answer the question using only the context "
        "provided below. If the answer is not contained in the context, reply that "
        "you do not have enough information.\n\n"
        f"Context:\n{context}"
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]

    print("\nAnswer:\n")
    stream = llm.chat.completions.create(
        model=FOUNDRY_MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading embedding model (first run downloads ~90 MB) ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Connecting to Qdrant at {QDRANT_URL} ...")
    qdrant = QdrantClient(url=QDRANT_URL)

    print(f"Connecting to Foundry Local at {FOUNDRY_BASE_URL} ...")
    llm = OpenAI(base_url=FOUNDRY_BASE_URL, api_key=FOUNDRY_API_KEY)

    # Ingest documents from the docs/ folder.
    print(f"\nIngesting documents from '{DOCS_FOLDER}' ...")
    count = ingest_documents(qdrant, embedder)
    if count:
        print(f"Stored {count} chunk(s) in Qdrant collection '{COLLECTION_NAME}'.\n")

    # Start the interactive question loop.
    print(
        f"RAG assistant ready.\n"
        f"  Vector DB : {QDRANT_URL}  (collection: {COLLECTION_NAME})\n"
        f"  LLM       : {FOUNDRY_BASE_URL}  (model: {FOUNDRY_MODEL})\n"
        f"Type 'exit' to quit.\n"
    )

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

        if question.lower() in ("exit", "quit", "q"):
            sys.exit(0)

        if not question:
            continue

        context = retrieve_context(qdrant, embedder, question)
        if not context:
            print("No relevant context found in the vector database.\n")
            continue

        generate_answer(llm, context, question)


if __name__ == "__main__":
    main()
