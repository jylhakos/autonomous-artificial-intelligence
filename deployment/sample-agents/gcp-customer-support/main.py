from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import support_agent
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Support Agent API",
    description="AI-powered customer support service",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message through the support agent."""
    try:
        logger.info(f"Processing message: {request.message}")
        result = support_agent.run(request.message)
        
        return ChatResponse(
            response=result.message,
            session_id=request.session_id
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "customer-support-agent"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
