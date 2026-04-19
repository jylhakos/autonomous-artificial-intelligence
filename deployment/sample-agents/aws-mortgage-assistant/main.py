from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from mangum import Mangum
import logging
from agent import mortgage_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mortgage Assistant Agent API",
    description="AI-powered mortgage advisory service",
    version="1.0.0"
)

security = HTTPBearer()

class AffordabilityRequest(BaseModel):
    annual_income: float
    monthly_debts: float
    down_payment: float

class RateRequest(BaseModel):
    credit_score: int
    loan_type: str = "conventional"

class QuestionRequest(BaseModel):
    question: str

@app.post("/calculate-affordability")
async def calculate_affordability(request: AffordabilityRequest):
    """Calculate home affordability."""
    try:
        result = mortgage_agent.calculate_affordability(
            annual_income=request.annual_income,
            monthly_debts=request.monthly_debts,
            down_payment=request.down_payment
        )
        return result
    except Exception as e:
        logger.error(f"Error calculating affordability: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-rates")
async def get_rates(
    request: RateRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get current mortgage rates (requires authentication)."""
    try:
        result = mortgage_agent.fetch_current_rates(
            credit_score=request.credit_score,
            loan_type=request.loan_type
        )
        return result
    except Exception as e:
        logger.error(f"Error fetching rates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about mortgages."""
    try:
        response = mortgage_agent.ask_question(request.question)
        return {"question": request.question, "answer": response}
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mortgage-agent"}

# Lambda handler
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
