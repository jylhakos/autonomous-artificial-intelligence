from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import tempfile
import os
from agent import document_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Analysis Agent API",
    description="Local AI-powered document analysis",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    top_k: int = 3

class SummaryRequest(BaseModel):
    document_id: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a PDF document."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        document = document_agent.load_pdf(tmp_path)
        
        os.unlink(tmp_path)
        
        if "error" in document:
            raise HTTPException(status_code=500, detail=document["error"])
        
        index_result = document_agent.index_document(document)
        
        return {
            "document_id": document["document_id"],
            "filename": document["filename"],
            "num_pages": document["num_pages"],
            "chunks": index_result.get("chunks_created", 0),
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about documents."""
    try:
        answer = document_agent.answer_question(
            question=request.question,
            document_id=request.document_id,
            top_k=request.top_k
        )
        return {
            "question": request.question,
            "answer": answer,
            "document_id": request.document_id
        }
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize_document(request: SummaryRequest):
    """Generate a document summary."""
    try:
        summary = document_agent.summarize_document(request.document_id)
        return {
            "document_id": request.document_id,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_document(request: SummaryRequest):
    """Analyze a document."""
    try:
        analysis = document_agent.analyze_document(request.document_id)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "document-analysis-agent"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
