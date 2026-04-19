# Mortgage Assistant Agent

This sample demonstrates a mortgage advisory AI agent deployed on Amazon Web Services using Bedrock and the AgentCore runtime.

## Features

- Home affordability calculations
- Current mortgage rate information
- Pre-qualification estimates
- Session-based security policies
- Integration with AWS Bedrock models

## Prerequisites

- AWS account with Bedrock access
- AWS CLI installed and configured
- Python 3.10 or later
- IAM permissions for Bedrock and Lambda

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS credentials:
```bash
aws configure
```

3. Set environment variables:
```bash
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
```

## Running Locally

```bash
python main.py
```

## Deploying to AWS Lambda

```bash
# Package Lambda function
pip install -t package -r requirements.txt
cd package && zip -r ../function.zip . && cd ..
zip -g function.zip agent.py main.py

# Deploy
aws lambda create-function \
    --function-name mortgage-agent \
    --runtime python3.11 \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/AgentLambdaRole \
    --handler main.handler \
    --zip-file fileb://function.zip
```

## API Usage

```bash
curl -X POST http://localhost:8000/calculate-affordability \
  -H "Content-Type: application/json" \
  -d '{"annual_income": 80000, "monthly_debts": 500, "down_payment": 40000}'
```

## Documentation

See the main deployment guide: [Amazon-AWS.md](../../Amazon-AWS.md)
