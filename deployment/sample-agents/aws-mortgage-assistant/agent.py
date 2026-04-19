import boto3
from typing import Dict, Any
import json
import os
from decimal import Decimal

bedrock_runtime = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))

class MortgageAgent:
    """Mortgage advisory agent with Bedrock integration."""
    
    def __init__(self, model_id: str = None):
        self.model_id = model_id or os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
        self.bedrock = bedrock_runtime
    
    def calculate_affordability(self, annual_income: float, monthly_debts: float, down_payment: float) -> Dict[str, Any]:
        """Calculate maximum affordable home price.
        
        Uses standard 28/36 rule for debt-to-income ratios.
        
        Args:
            annual_income: Annual gross income
            monthly_debts: Monthly debt payments
            down_payment: Available down payment amount
            
        Returns:
            Dictionary with affordability analysis
        """
        monthly_income = annual_income / 12
        
        # 28% rule: housing payment should not exceed 28% of monthly income
        max_housing_payment = monthly_income * 0.28
        
        # 36% rule: total debt should not exceed 36% of monthly income
        max_total_debt = monthly_income * 0.36
        max_housing_with_debt = max_total_debt - monthly_debts
        
        # Use the more conservative limit
        max_monthly_payment = min(max_housing_payment, max_housing_with_debt)
        
        # Estimate home price (30-year mortgage at 7% interest)
        rate_monthly = 0.07 / 12
        n_payments = 360
        
        if rate_monthly > 0:
            loan_amount = max_monthly_payment / (
                rate_monthly * (1 + rate_monthly)**n_payments / ((1 + rate_monthly)**n_payments - 1)
            )
        else:
            loan_amount = max_monthly_payment * n_payments
        
        max_home_price = loan_amount + down_payment
        
        return {
            "max_home_price": round(max_home_price, 2),
            "max_monthly_payment": round(max_monthly_payment, 2),
            "loan_amount": round(loan_amount, 2),
            "down_payment": down_payment,
            "dti_ratio": round((max_monthly_payment + monthly_debts) / monthly_income, 3),
            "disclaimer": "This is an estimate. Final approval subject to underwriting."
        }
    
    def fetch_current_rates(self, credit_score: int, loan_type: str = "conventional") -> Dict[str, Any]:
        """Fetch current mortgage rates.
        
        Args:
            credit_score: Customer credit score
            loan_type: Type of mortgage
            
        Returns:
            Dictionary with rate information
        """
        base_rate = 7.0
        
        if credit_score >= 760:
            rate_adjustment = -0.5
        elif credit_score >= 700:
            rate_adjustment = -0.25
        elif credit_score >= 640:
            rate_adjustment = 0
        else:
            rate_adjustment = 0.5
        
        current_rate = base_rate + rate_adjustment
        
        return {
            "loan_type": loan_type,
            "interest_rate": current_rate,
            "apr": current_rate + 0.125,
            "credit_score_used": credit_score,
            "rate_lock_days": 60,
            "disclaimer": "Rates subject to change. Final rate determined at underwriting."
        }
    
    def ask_question(self, question: str, authenticated: bool = False) -> str:
        """Process a question through the Bedrock model.
        
        Args:
            question: User's question
            authenticated: Whether user is authenticated
            
        Returns:
            Agent's response
        """
        system_prompt = """You are a knowledgeable mortgage advisory assistant.

Your capabilities:
- Calculate home affordability based on income and debts
- Provide current mortgage rates (requires authentication)
- Generate pre-qualification estimates
- Explain mortgage terms and processes

Guidelines:
- Always include required federal disclosures with rate quotes
- Never guarantee approval - explain that underwriting is required
- Be transparent about factors affecting rates
- Recommend speaking with a licensed loan officer for formal approval

Important compliance notes:
- All rate quotes must include: "Rates subject to change and final approval"
- Must disclose: "This is not a commitment to lend"
- Must inform customers of factors affecting final rate
"""
        
        messages = [{"role": "user", "content": question}]
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "temperature": 0.7,
            "system": system_prompt,
            "messages": messages
        }
        
        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        
        except Exception as e:
            return f"Error processing request: {str(e)}"

mortgage_agent = MortgageAgent()

if __name__ == "__main__":
    # Test the agent
    print("Mortgage Assistant Agent - Test Mode\n")
    print("=" * 50)
    
    # Test affordability calculation
    print("\n1. Affordability Calculation:")
    result = mortgage_agent.calculate_affordability(
        annual_income=80000,
        monthly_debts=500,
        down_payment=40000
    )
    print(json.dumps(result, indent=2))
    
    # Test rate fetch
    print("\n2. Current Rates:")
    rates = mortgage_agent.fetch_current_rates(
        credit_score=740,
        loan_type="conventional"
    )
    print(json.dumps(rates, indent=2))
    
    # Test question answering
    print("\n3. Question answering:")
    question = "What factors affect my mortgage rate?"
    response = mortgage_agent.ask_question(question)
    print(f"Q: {question}")
    print(f"A: {response}")
