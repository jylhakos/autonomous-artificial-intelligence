import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Dict, Any
import PyPDF2
import os
import hashlib
from pathlib import Path

class DocumentAnalysisAgent:
    """Local document analysis agent using Ollama."""
    
    def __init__(self, 
                 model: str = "qwen2.5:32b",
                 ollama_host: str = None,
                 chromadb_path: str = None):
        self.model = model
        self.ollama_host = ollama_host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.chromadb_path = chromadb_path or os.getenv('CHROMADB_PATH', './data/chromadb')
        
        os.environ['OLLAMA_HOST'] = self.ollama_host
        
        self.embeddings = OllamaEmbeddings(model=model)
        
        Path(self.chromadb_path).mkdir(parents=True, exist_ok=True)
        
        self.vectorstore = Chroma(
            persist_directory=self.chromadb_path,
            embedding_function=self.embeddings
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_pdf(self, file_path: str) -> Dict[str, Any]:
        """Load and parse a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document metadata and content
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
                
                return {
                    "document_id": doc_id,
                    "filename": os.path.basename(file_path),
                    "num_pages": num_pages,
                    "text": text,
                    "text_length": len(text)
                }
        
        except Exception as e:
            return {"error": f"Failed to load PDF: {str(e)}"}
    
    def index_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Index a document into the vector store.
        
        Args:
            document: Document to index
            
        Returns:
            Indexing results
        """
        try:
            text = document.get("text", "")
            doc_id = document.get("document_id")
            
            chunks = self.text_splitter.split_text(text)
            
            metadatas = [
                {
                    "document_id": doc_id,
                    "filename": document.get("filename", ""),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                for i in range(len(chunks))
            ]
            
            self.vectorstore.add_texts(
                texts=chunks,
                metadatas=metadatas
            )
            
            return {
                "document_id": doc_id,
                "chunks_created": len(chunks),
                "status": "indexed"
            }
        
        except Exception as e:
            return {"error": f"Failed to index document: {str(e)}"}
    
    def answer_question(self, question: str, document_id: str = None, top_k: int = 3) -> str:
        """Answer a question about documents.
        
        Args:
            question: Question to answer
            document_id: Optional specific document ID
            top_k: Number of context chunks to retrieve
            
        Returns:
            Answer text
        """
        try:
            filter_dict = {"document_id": document_id} if document_id else None
            
            results = self.vectorstore.similarity_search(
                question,
                k=top_k,
                filter=filter_dict
            )
            
            context = "\n\n".join([doc.page_content for doc in results])
            
            prompt = f"""Based on the following context from documents, answer the question.
If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful document analysis assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response['message']['content']
        
        except Exception as e:
            return f"Error answering question: {str(e)}"
    
    def summarize_document(self, document_id: str) -> str:
        """Generate a summary of a document.
        
        Args:
            document_id: Document to summarize
            
        Returns:
            Summary text
        """
        try:
            results = self.vectorstore.get(
                where={"document_id": document_id}
            )
            
            if not results or not results.get('documents'):
                return "Document not found"
            
            full_text = " ".join(results['documents'][:10])
            
            prompt = f"""Summarize the following document in a comprehensive yet concise manner.
Include:
1. Main topic and purpose
2. Key points (3-5 bullets)
3. Important conclusions or findings

Document text:
{full_text[:4000]}

Summary:"""

            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert document summarizer."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response['message']['content']
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def analyze_document(self, document_id: str) -> Dict[str, Any]:
        """Perform detailed analysis of a document.
        
        Args:
            document_id: Document to analyze
            
        Returns:
            Analysis results
        """
        try:
            results = self.vectorstore.get(
                where={"document_id": document_id}
            )
            
            if not results or not results.get('documents'):
                return {"error": "Document not found"}
            
            metadata = results['metadatas'][0] if results.get('metadatas') else {}
            text_sample = " ".join(results['documents'][:5])
            
            prompt = f"""Analyze this document and provide:
1. Document type (academic paper, report, manual, etc.)
2. Main topics covered
3. Writing style and tone
4. Target audience
5. Key themes

Document sample:
{text_sample[:2000]}

Provide analysis in JSON format."""

            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a document analysis expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return {
                "document_id": document_id,
                "filename": metadata.get("filename", ""),
                "total_chunks": metadata.get("total_chunks", 0),
                "analysis": response['message']['content']
            }
        
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

document_agent = DocumentAnalysisAgent()

if __name__ == "__main__":
    print("Document Analysis Agent - Test Mode\n")
    print("=" * 50)
    
    # Test with a sample text (simulating PDF)
    sample_doc = {
        "document_id": "test-001",
        "filename": "sample.pdf",
        "num_pages": 1,
        "text": """Artificial Intelligence in Healthcare

Introduction:
Artificial intelligence is transforming healthcare through improved diagnostics,
personalized treatment plans, and operational efficiency. This document explores
key applications and future directions.

Applications:
1. Medical imaging analysis
2. Drug discovery
3. Patient monitoring
4. Treatment planning

Conclusion:
AI has enormous potential to improve patient outcomes and reduce costs.""",
        "text_length": 500
    }
    
    print("\n1. Indexing Document:")
    index_result = document_agent.index_document(sample_doc)
    print(index_result)
    
    print("\n2. Answering Question:")
    answer = document_agent.answer_question(
        "What are the main applications of AI in healthcare?",
        document_id="test-001"
    )
    print(f"Answer: {answer}")
    
    print("\n3. Document Summary:")
    summary = document_agent.summarize_document("test-001")
    print(f"Summary: {summary}")
    
    print("\n4. Document Analysis:")
    analysis = document_agent.analyze_document("test-001")
    print(f"Analysis: {analysis}")
