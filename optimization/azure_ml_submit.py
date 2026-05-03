"""
azure_ml_submit.py
==================
Submits a fine-tuning job to Azure Machine Learning using the Azure ML SDK v2.

README section: "Azure Machine Learning for Production Workflows"

Concept covered:
    Azure ML provides managed compute clusters, a model registry, and managed
    online endpoints for production LLM workflows.  This script shows how to
    programmatically submit a command job that runs a fine-tuning script
    (`fine_tune.py`) on a GPU cluster using the Azure ML Python SDK v2.

    The fine_tune.py script (not included here) would typically:
    - Load a base model from Hugging Face or the Azure ML model registry
    - Apply LoRA / QLoRA adapters via Hugging Face PEFT
    - Train on a curated dataset stored in Azure ML datastore
    - Save the adapter weights to the output directory

Related README sections:
    - "Parameter-Efficient Fine-Tuning (PEFT) / LoRA"  — technique being applied
    - "Optimization on Microsoft Azure (Cloud)"        — Azure ecosystem overview
    - "Azure Container Apps for LoRA Fine-Tuning"      — alternative Azure path

Requirements:
    - Azure subscription with an Azure ML workspace
    - Virtual environment activated (see README "Python Virtual Environment Setup")

    pip install azure-ai-ml azure-identity
    # or:
    pip install -r requirements.txt

Usage:
    # 1. Activate the virtual environment
    source .venv/bin/activate

    # 2. Set environment variables (do NOT hard-code credentials in source files)
    export AZURE_SUBSCRIPTION_ID="<your-subscription-id>"
    export AZURE_RESOURCE_GROUP="<your-resource-group>"
    export AZURE_ML_WORKSPACE="<your-workspace-name>"

    # 3. Authenticate via the Azure CLI before running
    az login

    # 4. Run the script
    python azure_ml_submit.py
"""

import os
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

# ---------------------------------------------------------------------------
# Read connection details from environment variables
# Never hard-code subscription IDs, resource group names, or credentials.
# ---------------------------------------------------------------------------
subscription_id   = os.environ["AZURE_SUBSCRIPTION_ID"]
resource_group    = os.environ["AZURE_RESOURCE_GROUP"]
workspace_name    = os.environ["AZURE_ML_WORKSPACE"]

# ---------------------------------------------------------------------------
# Authenticate using DefaultAzureCredential
# This automatically picks up credentials from the Azure CLI, environment
# variables, managed identity, or interactive browser login (in that order).
# ---------------------------------------------------------------------------
print(f"Connecting to Azure ML workspace '{workspace_name}' ...")
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

# ---------------------------------------------------------------------------
# Define the training command job
# ---------------------------------------------------------------------------
job = command(
    code="./src",   # Local directory with training code to upload
    command=(
        "python fine_tune.py "
        "--model_name meta-llama/Llama-2-7b-hf "
        "--output_dir ./outputs"
    ),
    environment="azureml:AzureML-PyTorch-2.0-cuda11.8:1",
    compute="gpu-cluster",           # Name of the GPU compute cluster in Azure ML
    display_name="llm-lora-finetuning",
    description="QLoRA fine-tuning of Llama-2-7B on a GPU cluster",
)

# ---------------------------------------------------------------------------
# Submit the job
# ---------------------------------------------------------------------------
print("Submitting fine-tuning job to Azure ML ...")
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted successfully.")
print(f"Job name:   {returned_job.name}")
print(f"Job status: {returned_job.status}")
print(f"Studio URL: {returned_job.studio_url}")
