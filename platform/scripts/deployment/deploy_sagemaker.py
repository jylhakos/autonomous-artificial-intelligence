"""
AWS SageMaker Deployment: Deploy an MLflow Model to a SageMaker Endpoint
------------------------------------------------------------------------
Demonstrates how to build and push an MLflow model container to ECR and
deploy it as a real-time SageMaker inference endpoint.

Prerequisites:
    - AWS CLI configured (aws configure)
    - IAM role with SageMaker and ECR permissions
    - Docker installed and running
    - MLflow model registered in the tracking server

Run:
    python scripts/deployment/deploy_sagemaker.py

Requires (in activated virtual environment):
    pip install mlflow boto3
"""

import mlflow.sagemaker as mfs
import boto3

# --------------------------------------------------------------------------
# Configuration — replace these values with your own AWS environment details
# --------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "iris-classifier"
MODEL_VERSION = "1"
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

AWS_REGION = "us-east-1"
# SageMaker execution role ARN with AmazonSageMakerFullAccess policy
EXECUTION_ROLE_ARN = "arn:aws:iam::<ACCOUNT_ID>:role/<SAGEMAKER_ROLE>"
# ECR image built with: mlflow sagemaker build-and-push-container
ECR_IMAGE_URI = "<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:latest"
APP_NAME = "iris-classifier-endpoint"
# --------------------------------------------------------------------------


def build_and_push_container():
    """
    Build the MLflow SageMaker container and push it to ECR.
    Run this once before the first deployment.

    Equivalent CLI command:
        mlflow sagemaker build-and-push-container
    """
    print("Build and push container via CLI before deploying:")
    print("  mlflow sagemaker build-and-push-container")
    print(f"  (pushes to ECR region {AWS_REGION})\n")


def deploy_to_sagemaker():
    """Deploy the registered MLflow model to a SageMaker real-time endpoint."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"Deploying model '{MODEL_URI}' to SageMaker endpoint '{APP_NAME}'...")

    mfs.deploy(
        app_name=APP_NAME,
        model_uri=MODEL_URI,
        region_name=AWS_REGION,
        mode=mfs.DEPLOYMENT_MODE_CREATE,
        execution_role_arn=EXECUTION_ROLE_ARN,
        image_url=ECR_IMAGE_URI,
        instance_type="ml.m5.large",
        instance_count=1,
    )
    print(f"Endpoint '{APP_NAME}' deployment initiated.")
    print("Check status in the AWS SageMaker Console or with boto3.")


def check_endpoint_status():
    """Check whether the SageMaker endpoint is InService."""
    sm_client = boto3.client("sagemaker", region_name=AWS_REGION)
    response = sm_client.describe_endpoint(EndpointName=APP_NAME)
    status = response["EndpointStatus"]
    print(f"Endpoint '{APP_NAME}' status: {status}")
    return status


def invoke_endpoint(input_data: list):
    """Send a prediction request to the deployed SageMaker endpoint."""
    import json

    runtime_client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    payload = json.dumps({"instances": input_data})

    response = runtime_client.invoke_endpoint(
        EndpointName=APP_NAME,
        ContentType="application/json",
        Body=payload,
    )
    result = json.loads(response["Body"].read().decode())
    print(f"Prediction result: {result}")
    return result


if __name__ == "__main__":
    build_and_push_container()
    deploy_to_sagemaker()
    # After deployment completes (check status), run:
    # check_endpoint_status()
    # invoke_endpoint([[5.1, 3.5, 1.4, 0.2]])
