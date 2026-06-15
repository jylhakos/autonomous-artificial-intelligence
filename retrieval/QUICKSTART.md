# Quick Start Guide - RAG Pipeline

Get started with Retrieval-Augmented Generation in 5 minutes.

## Prerequisites

- Python 3.9 or higher
- OpenAI API key (or Azure OpenAI credentials)
- Weaviate running locally (via Docker) or Pinecone account

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up Environment Variables

Create a `.env` file in this directory:

```bash
# For OpenAI
OPENAI_API_KEY=your-openai-key-here

# OR for Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_LLM_DEPLOYMENT=gpt-4

# For Pinecone (optional)
PINECONE_API_KEY=your-pinecone-key
```

## Step 3: Start Weaviate (Local)

Using Docker:

```bash
docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  --name weaviate \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  -e ENABLE_MODULES='text2vec-openai' \
  -e OPENAI_APIKEY=$OPENAI_API_KEY \
  cr.weaviate.io/semitechnologies/weaviate:latest
```

Verify it's running:
```bash
curl http://localhost:8080/v1/.well-known/ready
```

## Step 4: Prepare Your Data

Create a `data/` directory and add some text files:

```bash
mkdir -p data
echo "Your document content here..." > data/sample.txt
```

Or use the provided sample data creator:

```bash
python create_sample_data.py
```

## Step 5: Run the Complete Pipeline

```bash
python complete_rag_pipeline.py
```

This will:
1. Load documents from `data/`
2. Chunk them using recursive chunking
3. Generate embeddings
4. Index into Weaviate
5. Run sample queries
6. Evaluate performance

## Step 6: Explore Individual Components

### Chunking with LangChain

```bash
python chunking_langchain.py
```

### Chunking with LlamaIndex + Azure

```bash
python chunking_llamaindex.py
```

### Evaluation Metrics

```bash
python rag_evaluation.py
```

## Common Commands

### Interactive Query

```python
from complete_rag_pipeline import RAGPipeline, RAGConfig

config = RAGConfig()
pipeline = RAGPipeline(config)
pipeline.initialize_models()
pipeline.connect_vector_db()

# Load and index your documents once
docs = pipeline.load_documents("data/")
chunks = pipeline.chunk_documents(docs)
pipeline.index_chunks(chunks)

# Now query interactively
while True:
    query = input("\nEnter your question (or 'quit'): ")
    if query.lower() == 'quit':
        break
    
    result = pipeline.query(query)
    print(f"\nAnswer: {result['answer']}")
    print(f"Latency: {result['latency_ms']:.2f}ms")

pipeline.close()
```

### Custom Chunking Strategy

```python
from chunking_langchain import fixed_size_chunking, semantic_chunking
from langchain_community.document_loaders import TextLoader

# Load document
loader = TextLoader("data/sample.txt")
docs = loader.load()

# Try different strategies
fixed_chunks = fixed_size_chunking(docs, chunk_size=256, chunk_overlap=25)
# semantic_chunks = semantic_chunking(docs, embeddings_model)

print(f"Fixed-size: {len(fixed_chunks)} chunks")
```

## Troubleshooting

### Weaviate Connection Error

```
Error: Failed to connect to Weaviate
```

**Solution**: Ensure Weaviate is running:
```bash
docker ps | grep weaviate
```

Restart if needed:
```bash
docker restart weaviate
```

### OpenAI API Error

```
Error: Incorrect API key provided
```

**Solution**: Verify your API key in `.env` and ensure it's loaded:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Empty Retrieval Results

```
Warning: No chunks retrieved
```

**Solution**: 
1. Verify chunks were indexed: Check Weaviate dashboard at http://localhost:8080
2. Try a broader query
3. Reduce `top_k` value

## Next Steps

1. **Tune Chunking**: Experiment with different chunk sizes and overlap values
2. **Add Metadata**: Enhance chunks with source, page numbers, timestamps
3. **Implement Hybrid Search**: Combine vector search with keyword matching
4. **Add Re-ranking**: Use cross-encoder models to re-score results
5. **Monitor Performance**: Track retrieval accuracy and latency metrics

## Resources

- [Microsoft - RAG Architecture](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase)
- [Weaviate - Chunking Strategies](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Pinecone - Chunking Best Practices](https://www.pinecone.io/learn/chunking-strategies/)
- [LlamaIndex - Vector Stores](https://developers.llamaindex.ai/python/framework/community/integrations/vector_stores/)

## Support

For issues or questions, refer to:
- Full documentation in [RAG.md](RAG.md)

