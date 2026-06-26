import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Define Paths and Constants
DOC_PATH = "document.txt"
DB_DIR = "./chroma_db"

print("🔄 Processing Iris dataset document...")

# 2. Load and Chunk Document
loader = TextLoader(DOC_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

print(f"✅ Document loaded and split into {len(chunks)} chunks")

# 3. Initialize Local Ollama Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Initialize and Populate Local Chroma Vector Database
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIR
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 5. Setup LLM and Prompt Template
llm = ChatOllama(model="llama3.2", temperature=0)

template = """
You are a helpful assistant. Answer the question based ONLY on the following context:
{context}

Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. Build the RAG Chain Pipeline
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Execute Local Query
query = "What are the morphological characteristics of Iris setosa?"
print(f"\n🔍 Querying local system: '{query}'")

response = rag_chain.invoke(query)
print("\n🤖 Llama 3.2 Response:")
print(response)
