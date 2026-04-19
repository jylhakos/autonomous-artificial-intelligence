from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from agent import research_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Assistant Agent API",
    description="AI-powered academic research assistant",
    version="1.0.0"
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class AnalyzeRequest(BaseModel):
    document: Dict[str, Any]

class SummaryRequest(BaseModel):
    topic: str
    documents: List[Dict[str, Any]]

class QuestionRequest(BaseModel):
    question: str
    context: Optional[str] = None

@app.post("/search")
async def search_documents(request: SearchRequest):
    """Search for academic documents."""
    try:
        results = research_agent.search_documents(
            query=request.query,
            top_k=request.top_k
        )
        return {"query": request.query, "results": results}
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_document(request: AnalyzeRequest):
    """Analyze a research document."""
    try:
        analysis = research_agent.analyze_document(request.document)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def generate_summary(request: SummaryRequest):
    """Generate a research summary."""
    try:
        summary = research_agent.generate_research_summary(
            topic=request.topic,
            documents=request.documents
        )
        return {"topic": request.topic, "summary": summary}
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a research question."""
    try:
        answer = research_agent.ask_question(
            question=request.question,
            context=request.context
        )
        return {"question": request.question, "answer": answer}
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "research-agent"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
