from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph

# Neo4j Credentials
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

# 1. Initialize local embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Extract and split text from Iris dataset document
loader = TextLoader("document.txt")
documents = loader.load()
text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# 3. Store documents as a vector index inside Neo4j
vector_store = Neo4jVector.from_documents(
    docs,
    embeddings,
    url=NEO4J_URL,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    index_name="document_vector_index",
    node_label="DocumentChunk",
    embedding_node_property="embedding",
    text_node_property="text"
)

# 4. Connect to structured graph interface
graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USER, password=NEO4J_PASSWORD)
print("Data successfully ingested and Neo4j graph connected!")
