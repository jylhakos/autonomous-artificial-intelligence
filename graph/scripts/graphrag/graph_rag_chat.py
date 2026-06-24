import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphQAChain

# 1. Configuration & Connection Setup
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

print("Connecting to Neo4j database...")
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

# Initialize local LLM via Ollama
llm = OllamaLLM(model="llama3.2", temperature=0)

# 2. Load and Split document.txt
if not os.path.exists("document.txt"):
    print("Error: Please place your 'document.txt' file in this directory.")
    exit(1)

print("Loading document.txt...")
loader = TextLoader("document.txt")
documents = loader.load()

# Split text into manageable tokens for local LLM processing
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
docs = text_splitter.split_documents(documents)

# 3. Extract Knowledge Graph Entities & Relations
print("Extracting entities and relationships using Llama 3.2...")
# You can strictly define allowed nodes/edges to make the graph cleaner
llm_transformer = LLMGraphTransformer(llm=llm)

print("Converting text documents into graph data structure...")
graph_docs = llm_transformer.convert_to_graph_documents(docs)

# Store extracted information into Neo4j
print(f"Ingesting {len(graph_docs)} graph documents into Neo4j...")
graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
print("Ingestion complete! Check your Neo4j browser to view nodes and edges.")

# 4. Create GraphRAG Query Chain
# GraphQAChain automatically converts human questions into Cypher queries using the LLM
qa_chain = GraphQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

# 5. Interactive Chat Loop
print("\n--- GraphRAG Chat initialized. Type 'exit' to quit ---")
while True:
    user_query = input("\nYou: ")
    if user_query.strip().lower() == 'exit':
        break
    
    if not user_query.strip():
        continue
        
    try:
        print("Thinking...")
        response = qa_chain.invoke({"query": user_query})
        print(f"\nAI: {response['result']}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
