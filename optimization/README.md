# Optimization of Machine Learning and Large Language Models

This document explores the concepts, techniques, tools, and workflows for optimizing machine learning (ML) models and large language models (LLMs), including local and cloud-based approaches. It includes step-by-step setup instructions for a Python environment on Linux with VS Code and practical code examples.

---

## Table of Contents

1. [What are Large Language Models (LLMs)?](#what-are-large-language-models-llms)
2. [Model Size and Key Concepts](#model-size-and-key-concepts)
3. [Before Optimization: Baseline Metrics](#before-optimization-baseline-metrics)
4. [Optimization Techniques](#optimization-techniques)
   - [Quantization](#1-quantization-lowering-precision)
   - [Pruning](#2-pruning-removing-parameters)
   - [Knowledge Distillation](#3-knowledge-distillation-model-compression)
   - [Low-Rank Factorization](#4-low-rank-factorization)
   - [Parameter-Efficient Fine-Tuning (PEFT) and LoRA](#5-parameter-efficient-fine-tuning-peft--lora)
5. [Optimization Algorithms](#optimization-algorithms)
6. [Inference Optimization Techniques](#inference-optimization-techniques)
7. [Fine-Tuning Language Models](#fine-tuning-language-models)
8. [Key SDKs and Libraries for Optimization](#key-sdks-and-libraries-for-optimization)
9. [Local Environment Setup on Linux with VS Code](#local-environment-setup-on-linux-with-vs-code)
10. [Python Virtual Environment Setup](#python-virtual-environment-setup)
11. [Example Usage: 4-Bit Quantization with Hugging Face](#example-usage-4-bit-quantization-with-hugging-face)
12. [Example Usage: Few-Shot Prompt Optimization](#example-usage-few-shot-prompt-optimization)
13. [Example Usage: RAG for Accurate Knowledge Retrieval](#example-usage-rag-for-accurate-knowledge-retrieval)
14. [Optimization on Local Environment](#optimization-on-local-environment)
15. [Optimization on Microsoft Azure (Cloud)](#optimization-on-microsoft-azure-cloud)
16. [Microsoft Olive Optimization Workflow](#microsoft-olive-optimization-workflow)
17. [RAG Optimization with Azure AI Search](#rag-optimization-with-azure-ai-search)
18. [Optimization on AWS and Google Cloud](#optimization-on-aws-and-google-cloud)
19. [Summary: Choosing the Right Approach](#summary-choosing-the-right-approach)
20. [References](#references)

---

## What are Large Language Models (LLMs)?

Large language models (LLMs) are advanced AI systems that understand and generate natural language using deep learning techniques, specifically the transformer architecture. They are trained on massive text corpora and can perform tasks such as text generation, summarization, translation, code generation, and question answering.

LLMs rely on the transformer architecture, introduced in the 2018 paper "Attention is All You Need" by Vaswani et al. Transformers use self-attention mechanisms to weigh the significance of different tokens in a sequence, regardless of their position, enabling them to capture complex linguistic patterns.

The training process for LLMs consists of three main stages:

1. **Data collection**: The model is exposed to large volumes of text from diverse sources.
2. **Pre-training**: The model learns language representations through unsupervised tasks such as next-token prediction.
3. **Fine-tuning**: The model is further trained on a smaller, task-specific dataset to refine its performance.

Key LLM families include the GPT series (OpenAI), BERT (Google), RoBERTa (Facebook AI), T5 (Google Research), and the Microsoft Phi series.

---

## Model Size and Key Concepts

Machine learning model size is primarily defined by the number of trainable parameters (weights and biases) in a neural network, which directly influences capacity, memory footprint, and compute requirements.

### Parameters

Parameters are the internal variables (weights and biases) that a model adjusts during training. Larger parameter counts allow models to learn more intricate patterns but demand more memory and computational power.

- Small language models (SLMs): fewer than 10 billion parameters (e.g., Microsoft Phi-3 mini: 3.8B, Phi-3 small: 7B)
- Large language models (LLMs): more than 10 billion parameters (e.g., Llama-2-70B, Falcon-40B)

### Model Architecture

The transformer architecture determines model size through:

- Number of layers (transformer blocks)
- Number of attention heads per layer
- Hidden dimension size $d_{model}$
- Feed-forward network dimension $d_{ff}$

The total number of parameters in a transformer approximates:

$$N \approx 12 \cdot n_{layers} \cdot d_{model}^2$$

where $n_{layers}$ is the number of layers and $d_{model}$ is the hidden dimension.

### Memory Footprint

The VRAM required to load a model for inference depends on the parameter count and the numerical precision used to store each parameter:

$$\text{Memory (bytes)} = N_{params} \times \text{bytes per parameter}$$

| Precision | Bytes per Parameter | 7B Model VRAM |
|-----------|-------------------|---------------|
| FP32      | 4                 | ~28 GB        |
| FP16/BF16 | 2                 | ~14 GB        |
| INT8      | 1                 | ~7 GB         |
| INT4      | 0.5               | ~3.5 GB       |

### Context Window

The context window (or context length) is the maximum number of tokens a model can process in a single forward pass. Larger context windows increase memory demands, since the key-value (KV) cache grows linearly with context length and batch size.

---

## Before Optimization: Baseline Metrics

Before applying any optimization technique, establish a baseline by measuring:

- **Model accuracy**: Task-specific metrics such as perplexity, BLEU score, accuracy, or F1 score
- **Latency**: Time-to-first-token (TTFT) and tokens-per-second (TPS)
- **Memory usage**: Peak VRAM and RAM consumption
- **GPU or CPU utilization**: Hardware utilization during inference and training

These measurements allow you to quantify the trade-offs introduced by each optimization technique and ensure that model quality remains acceptable.

---

## Optimization Techniques

### 1. Quantization (Lowering Precision)

Quantization reduces the numerical precision of model weights and activations, typically converting floating-point values (FP32) to lower-precision formats such as FP16, INT8, or INT4. This reduces the memory footprint and speeds up inference, often with minimal loss in accuracy.

The core idea is to map a range of floating-point values $[x_{min}, x_{max}]$ to a discrete integer range $[-2^{b-1}, 2^{b-1} - 1]$ for $b$-bit signed integers:

$$x_{quantized} = \text{round}\left(\frac{x - x_{min}}{x_{max} - x_{min}} \cdot (2^b - 1)\right)$$

The scale factor $s$ and zero-point $z$ used for dequantization are:

$$s = \frac{x_{max} - x_{min}}{2^b - 1}, \quad z = -\text{round}\left(\frac{x_{min}}{s}\right)$$

**Types of quantization:**

- **Post-Training Quantization (PTQ)**: Applies quantization after the model is fully trained. Requires a small calibration dataset to estimate the range of activations. Faster to apply but may introduce slightly more accuracy loss.
- **Quantization-Aware Training (QAT)**: Simulates low-precision arithmetic during training using straight-through estimators for the rounding operation. The model adapts to quantization noise, resulting in better accuracy than PTQ.

**Quantization formats commonly used:**

- FP32 to FP16 or BF16: Half the memory, minimal accuracy loss, widely supported on GPUs
- FP16 to INT8: Further reduction, used in deployment with TensorRT, ONNX Runtime
- FP16 to INT4: Maximum compression (e.g., bitsandbytes NF4, GPTQ, AWQ), suitable for consumer GPUs

Quantizing from FP16 to INT4 cuts the model size from approximately 16 GB to about 4 GB for a 7B parameter model, increasing inference speed in the process.

---

### 2. Pruning (Removing Parameters)

Pruning identifies and removes non-critical components of a neural network, resulting in a sparser model with reduced compute and memory requirements.

When a model is trained, it builds a map of semantic connections. These connections, called parameters, gain or lose importance as more training data is introduced. Pruning removes the least important parameters based on a saliency criterion such as weight magnitude.

**Types of pruning:**

- **Unstructured Pruning**: Sets individual weights to zero based on a threshold (e.g., $|w| < \epsilon$). Produces a sparse weight matrix but requires specialized sparse compute libraries to achieve real speedups.
- **Structured Pruning**: Removes entire neurons, attention heads, channels, or layers. More compatible with standard hardware as it reduces the actual tensor dimensions.
- **Pruning-Aware Training**: Pruning is incorporated into the training recipe, performing model-wide scans of weights during training at a higher computational cost.

**The Lottery Ticket Hypothesis** (Frankle and Carlin, 2019): A randomly initialized, dense neural network contains a small subnetwork (a "winning lottery ticket") that, when trained in isolation, can match the full network's performance. This subnetwork can be identified by iterative magnitude pruning.

A simple magnitude-based pruning mask $m$ for a weight $w_i$:

$$m_i = \begin{cases} 0 & \text{if } |w_i| < \theta \\ 1 & \text{otherwise} \end{cases}$$

where $\theta$ is the pruning threshold controlling the sparsity level.

---

### 3. Knowledge Distillation (Model Compression)

Knowledge distillation trains a smaller student model to replicate the behavior of a larger, pre-trained teacher model. The student learns from the teacher's output probability distributions (soft targets) rather than just the hard ground-truth labels.

**Loss function for distillation:**

$$\mathcal{L}_{KD} = (1 - \alpha) \cdot \mathcal{L}_{CE}(y, \hat{y}_{student}) + \alpha \cdot T^2 \cdot \text{KL}\left(\sigma\!\left(\frac{z_T}{T}\right) \bigg\| \sigma\!\left(\frac{z_S}{T}\right)\right)$$

where:
- $\mathcal{L}_{CE}$ is the standard cross-entropy loss with ground-truth labels
- $z_T$ and $z_S$ are the teacher and student logits
- $T$ is the temperature parameter that softens the probability distributions
- $\alpha$ is the interpolation coefficient between the two losses
- $\sigma(\cdot)$ is the softmax function

**Key approaches:**

- **Soft Targets**: The student is trained to match the teacher's probability distribution over all classes, preserving rich inter-class relationship information.
- **Synthetic Data Training**: The teacher generates training data for the student. This approach was used in models like Alpaca, where a large model (e.g., GPT-4) generates instruction-following data to train a smaller model.

---

### 4. Low-Rank Factorization

This technique decomposes large weight matrices into smaller, lower-rank approximating matrices. For a weight matrix $W \in \mathbb{R}^{m \times n}$ with rank $r < \min(m, n)$, the approximation is:

$$W \approx U \cdot V^T, \quad U \in \mathbb{R}^{m \times r},\; V \in \mathbb{R}^{n \times r}$$

The number of parameters is reduced from $m \times n$ to $r \times (m + n)$. When $r \ll \min(m, n)$, this yields a significant parameter reduction while retaining the model's ability to represent complex relationships.

---

### 5. Parameter-Efficient Fine-Tuning (PEFT) / LoRA

**LoRA (Low-Rank Adaptation)** freezes the pre-trained model weights and injects small trainable rank-decomposition matrices into each transformer layer. Instead of updating the full weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA constrains the update to a low-rank decomposition:

$$W = W_0 + \Delta W = W_0 + B \cdot A$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and the rank $r \ll \min(d, k)$.

During training, $W_0$ is frozen and only $A$ and $B$ are updated. At initialization, $A$ is drawn from a random Gaussian distribution and $B$ is initialized to zero, so $\Delta W = 0$ at the start of training. The effective weight update is scaled by $\frac{\alpha}{r}$, where $\alpha$ is a hyperparameter.

The total number of trainable parameters per layer is $r \cdot (d + k)$ instead of $d \cdot k$. For a 7B parameter model, LoRA can reduce trainable parameters to less than 1% of the original.

**Key insight:** The rank parameter $r$ determines the size of the adapter layers. Higher ranks provide greater capacity for learning new patterns but also increase memory and compute requirements.

**QLoRA (Quantized LoRA)** extends LoRA by storing the frozen base model weights in 4-bit precision (NF4 format) instead of 16-bit. The 4-bit quantized weights are dequantized to BF16 on the fly during the forward pass for computation, while gradients flow through only the LoRA adapters. This allows training of 65B+ parameter models on a single 48GB GPU.

**PEFT (Parameter-Efficient Fine-Tuning)** is the Hugging Face library that implements LoRA, QLoRA, prefix tuning, prompt tuning, and other adapter-based methods. It enables modifying only a tiny subset of parameters during adaptation, making it practical to run specialized large models in low-memory environments.

---

## Optimization Algorithms

### AdamW (Adaptive Moment Estimation with Decoupled Weight Decay)

AdamW is the standard optimizer for LLM pre-training and fine-tuning. It maintains per-parameter adaptive learning rates based on first and second moment estimates of the gradient, with weight decay applied directly to the parameters rather than to the gradient update.

The update rule for parameter $\theta_t$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_{t-1}$$

where $\eta$ is the learning rate, $\lambda$ is the weight decay coefficient, and $\beta_1, \beta_2$ are the moment decay rates (typically 0.9 and 0.999).

AdamW handles sparse gradients well and its decoupled weight decay improves regularization compared to the original Adam optimizer.

### Gradient Descent Variations

- **Stochastic Gradient Descent (SGD)**: Updates weights using the gradient of the loss with respect to a single sample or mini-batch.
- **Mini-Batch Gradient Descent**: Standard practice in deep learning; computes gradients over a batch of examples.
- **Gradient Checkpointing**: Trades compute for memory by recomputing activations during the backward pass instead of storing them all.

### PPO (Proximal Policy Optimization)

PPO is used in Reinforcement Learning from Human Feedback (RLHF) to align LLMs with human preferences. It constrains the policy update to remain close to the previous policy using a clipping mechanism:

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ is the probability ratio, $\hat{A}_t$ is the advantage estimate, and $\epsilon$ is the clipping threshold.

### Neural Architecture Search (NAS)

NAS uses automated search algorithms to discover efficient model architectures and hyperparameters, reducing the manual effort of architecture design. LLMs themselves can be used as optimizers within NAS pipelines.

---

## Inference Optimization Techniques

### PagedAttention

PagedAttention manages key-value (KV) cache memory similarly to how operating systems manage virtual memory using paging. Instead of pre-allocating contiguous memory blocks for the full context, it allocates non-contiguous memory pages on demand. This significantly reduces memory waste and allows larger batch sizes during inference. PagedAttention is the core innovation of the vLLM serving engine.

### Grouped-Query Attention (GQA)

GQA is a middle ground between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). In MHA, each attention head has its own key and value projections. In MQA, all heads share a single key-value pair. GQA groups multiple query heads to share a single key-value head:

$$\text{GQA}(Q, K, V) = \text{Concat}\left(\text{head}_1, \ldots, \text{head}_{h_q}\right) W^O$$

where the $h_q$ query heads are divided into $g$ groups, and each group shares one of the $h_{kv} = g$ key-value heads. GQA maintains model quality close to MHA while achieving the memory efficiency of MQA.

### KV Caching

During autoregressive generation, the key and value matrices for previously generated tokens are cached and reused in subsequent decoding steps. This avoids redundant computation for the prefix of the sequence:

$$\text{Memory for KV cache} = 2 \times n_{layers} \times d_{model} \times n_{seq} \times \text{bytes per value}$$

### Continuous Batching

Rather than processing requests in fixed batches, continuous batching inserts new requests into the batch as slots become available when previous requests finish. This keeps GPU utilization high and reduces average latency.

### KleidiAI Micro-Kernels

Arm KleidiAI provides hand-optimized assembly micro-kernels for LLM inference on Arm CPUs. Combined with TorchAO quantization, these kernels provide significant performance improvements for CPU-based inference.

---

## Fine-Tuning Language Models

Fine-tuning involves further training a pre-trained LLM on a task-specific dataset (a transfer learning process). A pre-trained model has already learned a large amount of general information, and fine-tuning specializes it for a particular domain or task.

### The Memory Challenge in Fine-Tuning

During fine-tuning, memory is the primary bottleneck. Unlike inference, which only generates text, fine-tuning must store:

- Model weights
- Optimizer states (e.g., two moment tensors per parameter for Adam, doubling or tripling memory vs. inference)
- Gradients
- Activations (for backpropagation)

The total memory for full fine-tuning with mixed precision and AdamW:

$$\text{Memory} \approx (2 + 2 + 4 + 4) \times N_{params} \text{ bytes} = 16 \text{ bytes/param}$$

For a 7B model, this is approximately 112 GB, far beyond most consumer GPU capacities.

### Fine-Tuning Methods

| Method | Description | Memory Cost |
|--------|-------------|-------------|
| Full Fine-Tuning | Updates all model parameters | Very High |
| PEFT / LoRA | Freezes base model, trains small adapter matrices | Low |
| QLoRA | 4-bit quantized base + LoRA adapters | Very Low |

### Batching and Throughput

Batching groups multiple inputs together for simultaneous GPU processing, maximizing hardware utilization. Instead of processing one prompt at a time, the GPU processes multiple prompts in parallel. Continuous batching dynamically inserts new requests as old ones complete.

### Fine-Tuning Step-by-Step

1. **Data preparation**: Collect, clean, format, and split data into train/validation/test sets.
2. **Choose an approach**: Full fine-tuning, PEFT (LoRA/QLoRA), or instruction tuning.
3. **Set hyperparameters**: Configure learning rate, batch size, number of epochs, LoRA rank $r$, and scaling factor $\alpha$.
4. **Train the model**: Feed training data, monitor validation loss, adjust as needed.
5. **Evaluate performance**: Use task-relevant metrics on the held-out test set.
6. **Deploy**: Export the model and adapter weights for production use.

---

## Key SDKs and Libraries for Optimization

| Library / SDK | Purpose | Best For |
|---------------|---------|---------|
| vLLM | High-throughput inference with PagedAttention | Production serving on NVIDIA GPU |
| TensorRT-LLM (NVIDIA) | Maximum NVIDIA GPU performance via kernel fusion and quantization | NVIDIA GPU deployment |
| NVIDIA NeMo | Large-scale training with FP8, distributed training | Pre-training and fine-tuning at scale |
| DeepSpeed (Microsoft) | Distributed training, memory optimization (ZeRO) | Multi-GPU / multi-node training |
| Hugging Face PEFT | LoRA, QLoRA, prefix tuning | Efficient fine-tuning |
| Hugging Face Transformers | General-purpose model loading, inference, fine-tuning | Broad compatibility |
| bitsandbytes | 4-bit and 8-bit quantization for Hugging Face models | Low-VRAM inference |
| torchtune | PyTorch-native fine-tuning recipes for LLMs | Fine-tuning with PyTorch |
| TorchAO | PyTorch quantization and sparsity | CPU and GPU quantization |
| Unsloth | Fast fine-tuning for Llama, Mistral | Speed and memory efficiency |
| Llamafile | Single-executable LLM for CPU inference | Easy local deployment |
| Microsoft Olive | End-to-end optimization pipeline (quantization, graph opt.) | ONNX/DirectML/NPU deployment |
| llama.cpp | Efficient C/C++ inference for GGUF format models | Edge and on-premises deployment |

---

## Local Environment Setup on Linux with VS Code

### 1. Install VS Code on Linux

```bash
# Download and install VS Code (Debian/Ubuntu)
sudo apt update
sudo apt install software-properties-common apt-transport-https wget -y
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | \
  sudo tee /usr/share/keyrings/packages.microsoft.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/packages.microsoft.gpg] \
  https://packages.microsoft.com/repos/code stable main" | \
  sudo tee /etc/apt/sources.list.d/vscode.list
sudo apt update
sudo apt install code -y
```

### 2. Install Recommended VS Code Extensions

Open VS Code and install the following extensions:

- **Python** (ms-python.python): Python language support, IntelliSense, linting
- **Pylance** (ms-python.vscode-pylance): Fast Python language server
- **Jupyter** (ms-toolsai.jupyter): Notebook support for experimentation
- **GitLens** (eamodio.gitlens): Enhanced Git integration
- **AI Toolkit for VS Code** (ms-windows-ai-studio.windows-ai-studio): Fine-tune and deploy models locally

```bash
# Install extensions from the command line
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension eamodio.gitlens
```

### 3. Install System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python build tools and pip
sudo apt install python3 python3-pip python3-venv python3-dev -y

# Install Git and Git LFS (for downloading large model files)
sudo apt install git git-lfs -y
git lfs install

# Install build essentials (needed for some native Python packages)
sudo apt install build-essential gcc g++ cmake -y

# Optional: Install CUDA toolkit if you have an NVIDIA GPU
# Check NVIDIA driver installation first
nvidia-smi

# Install CUDA toolkit (adjust version to match your driver)
# sudo apt install nvidia-cuda-toolkit -y
```

### 4. Verify Python Installation

```bash
python3 --version
pip3 --version
```

---

## Python Virtual Environment Setup

It is mandatory to create and activate a Python virtual environment before installing any libraries or executing scripts. This isolates project dependencies and prevents conflicts with system Python packages.

### Create the Virtual Environment

```bash
# Navigate to the optimization project directory
cd /path/to/optimization

# Create a virtual environment named .venv
python3 -m venv .venv
```

### Activate the Virtual Environment

```bash
# Activate on Linux / macOS
source .venv/bin/activate

# Your terminal prompt should now show (.venv)
# Verify the correct Python interpreter is active
which python
python --version
```

### Select the Virtual Environment in VS Code

1. Open the Command Palette: `Ctrl+Shift+P`
2. Type `Python: Select Interpreter`
3. Choose the interpreter at `./.venv/bin/python`

### Install Required Libraries

After activating the virtual environment, install the required packages:

```bash
# Upgrade pip first
pip install --upgrade pip

# Core ML and LLM libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hugging Face ecosystem
pip install transformers datasets accelerate peft

# 4-bit and 8-bit quantization
pip install bitsandbytes

# Sentence embeddings (for RAG examples)
pip install sentence-transformers faiss-cpu

# Additional tools
pip install scipy numpy pandas tqdm

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Deactivate the Virtual Environment

```bash
deactivate
```

### Save and Restore Dependencies

```bash
# Save current dependencies to requirements.txt
pip freeze > requirements.txt

# Restore from requirements.txt in a new environment
pip install -r requirements.txt
```

---

## Practical Example Scripts

All practical examples in this project are provided as standalone Python scripts
that can be run directly from a terminal.  The same code is also shown inline in
the relevant sections below so that you can read the explanation and the
implementation side by side.

**Before running any script, create and activate the virtual environment:**

```bash
# Create the virtual environment (one-time setup)
python3 -m venv .venv

# Activate on Linux / macOS
source .venv/bin/activate

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Script Overview

| Script | What it does | Related README section |
|--------|-------------|------------------------|
| `quantization_example.py` | Loads Llama-2-7B in 4-bit NF4 format using bitsandbytes and runs a single inference call. Shows how a ~14 GB FP16 model fits into ~3.5 GB VRAM. | Example Usage: 4-Bit Quantization with Hugging Face |
| `prompt_optimization.py` | Demonstrates few-shot prompting for sentiment classification using GPT-2. Compares few-shot output against zero-shot output. No GPU required. | Example Usage: Few-Shot Prompt Optimization |
| `rag_example.py` | Builds a local FAISS vector index, encodes a small knowledge base with all-MiniLM-L6-v2, retrieves top-k relevant passages for a query, and constructs the augmented prompt. CPU-friendly. | Example Usage: RAG for Accurate Knowledge Retrieval |
| `fast_finetune.py` | Sets up a Llama-3-8B model in 4-bit QLoRA configuration with Unsloth, injects LoRA adapters into Q/K/V/O projections, and prints trainable parameter counts. Requires NVIDIA GPU (16 GB+ VRAM). | Using Unsloth for Fast Fine-Tuning |
| `azure_ml_submit.py` | Submits a fine-tuning command job to Azure Machine Learning using the Azure ML SDK v2. Reads Azure credentials from environment variables. Requires an Azure ML workspace. | Azure Machine Learning for Production Workflows |
| `azure_rag_search.py` | Production RAG pipeline using Azure AI Search for semantic retrieval and Azure OpenAI GPT-4o for grounded answer generation. Requires Azure AI Search and Azure OpenAI resources. | RAG Optimization with Azure AI Search |

### How to Run Each Script

```bash
# Activate virtual environment first
source .venv/bin/activate

# 4-bit quantization demo (requires CUDA GPU + Hugging Face account)
python quantization_example.py

# Few-shot prompt optimization (CPU-friendly)
python prompt_optimization.py

# Local RAG pipeline with FAISS (CPU-friendly)
python rag_example.py

# Fast LoRA fine-tuning with Unsloth (requires CUDA GPU)
# Install Unsloth first: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python fast_finetune.py

# Azure ML fine-tuning job submission (requires Azure subscription)
export AZURE_SUBSCRIPTION_ID="<subscription-id>"
export AZURE_RESOURCE_GROUP="<resource-group>"
export AZURE_ML_WORKSPACE="<workspace-name>"
python azure_ml_submit.py

# Azure AI Search + OpenAI RAG query (requires Azure resources)
export AZURE_SEARCH_ENDPOINT="https://<service>.search.windows.net"
export AZURE_SEARCH_KEY="<key>"
export AZURE_SEARCH_INDEX="<index>"
export AZURE_OPENAI_ENDPOINT="https://<endpoint>.openai.azure.com/"
export AZURE_OPENAI_KEY="<key>"
python azure_rag_search.py
```

---

## Example Usage: 4-Bit Quantization with Hugging Face

This example demonstrates loading a Hugging Face model in 4-bit precision using `bitsandbytes`. The technique allows a Llama-2-70B model, which would normally require high VRAM, to be served on fewer GPUs with minimal accuracy loss.

The code below is also available as the standalone script `quantization_example.py`.
Run it with `python quantization_example.py` after activating the virtual environment.
See the [Practical Example Scripts](#practical-example-scripts) section for setup instructions.

**Activate the virtual environment first:**

```bash
source .venv/bin/activate
```

Create a file named `quantization_example.py`:

```python
# Example of loading a model with 4-bit quantization using Hugging Face
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "meta-llama/Llama-2-7b-hf"

# Configure 4-bit quantization with NF4 format
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NormalFloat4 data type
    bnb_4bit_compute_dtype=torch.float16  # Compute in FP16 after dequantization
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"   # Automatically distribute across available GPUs/CPU
)

# Run inference
inputs = tokenizer("Explain the concept of quantization in machine learning:", return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Run the script:

```bash
python quantization_example.py
```

**Result**: A 7B model stored in FP16 (~14 GB VRAM) is reduced to approximately 3.5 GB in 4-bit NF4 format, enabling it to run on consumer-grade GPUs (8-12 GB VRAM).

---

## Example Usage: Few-Shot Prompt Optimization

Prompt optimization improves accuracy without any model training by providing the model with examples of the desired input-output format.

The code below is also available as the standalone script `prompt_optimization.py`.
Run it with `python prompt_optimization.py` after activating the virtual environment.
See the [Practical Example Scripts](#practical-example-scripts) section for setup instructions.

**Activate the virtual environment first:**

```bash
source .venv/bin/activate
```

Create a file named `prompt_optimization.py`:

```python
from transformers import pipeline

# Load a small model for demonstration
generator = pipeline("text-generation", model="gpt2", max_new_tokens=10)

# Few-shot prompt: provide examples to guide the model output format
few_shot_prompt = """Classify the sentiment of the following sentences.

Sentence: "This product is terrible."
Sentiment: Negative

Sentence: "Absolutely loved it!"
Sentiment: Positive

Sentence: "It was okay, nothing special."
Sentiment: Neutral

Sentence: "Okay."
Sentiment:"""

output = generator(few_shot_prompt, do_sample=False)
print(output[0]["generated_text"])
```

Run the script:

```bash
python prompt_optimization.py
```

**Result**: The model follows the established format and predicts "Neutral" rather than generating unrelated text. This technique (Chain-of-Thought prompting or few-shot prompting) requires no training and is effective for many classification and reasoning tasks.

---

## Example Usage: RAG for Accurate Knowledge Retrieval

Retrieval-Augmented Generation (RAG) inserts relevant context from an external knowledge base into the prompt, reducing hallucinations and improving accuracy for domain-specific questions.

The code below is also available as the standalone script `rag_example.py`.
Run it with `python rag_example.py` after activating the virtual environment.
See the [Practical Example Scripts](#practical-example-scripts) section for setup instructions.

**Activate the virtual environment first:**

```bash
source .venv/bin/activate
pip install faiss-cpu sentence-transformers
```

Create a file named `rag_example.py`:

```python
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Step 1: Define a small knowledge base (in production, this would be a large document corpus)
documents = [
    "Quantization reduces model precision from FP32 to INT8 or INT4.",
    "LoRA fine-tunes models by injecting small trainable adapter matrices.",
    "Pruning removes low-importance weights from a neural network.",
    "Knowledge distillation trains a small student model from a large teacher model.",
    "PagedAttention manages KV cache memory for efficient LLM serving.",
]

# Step 2: Encode documents into vector embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedding_model.encode(documents, convert_to_numpy=True)

# Step 3: Build a FAISS index for fast similarity search
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

def retrieve(query: str, top_k: int = 2) -> list[str]:
    """Retrieve the top_k most relevant documents for a query."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

# Step 4: Use retrieved context in a prompt
user_question = "How does LoRA reduce memory usage during fine-tuning?"
retrieved_docs = retrieve(user_question, top_k=2)

# Construct the augmented prompt
context = "\n".join(f"- {doc}" for doc in retrieved_docs)
augmented_prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {user_question}
Answer:"""

print("Augmented prompt sent to the LLM:")
print(augmented_prompt)
# In a full implementation, pass augmented_prompt to your LLM for generation
```

Run the script:

```bash
python rag_example.py
```

**Result**: Instead of relying on the base model's potentially outdated training data, RAG inserts the two most relevant knowledge base passages into the prompt, grounding the model's answer in factual, up-to-date information.

---

## Optimization on Local Environment

Running and optimizing LLMs locally requires careful selection of tools based on available hardware.

### Using llama.cpp for CPU / Edge Inference

llama.cpp is an optimized C/C++ framework that runs quantized models in GGUF format. It supports CPU inference without requiring a GPU and can offload layers to the GPU when available.

```bash
# Install build dependencies
sudo apt install git cmake build-essential -y

# Clone and build llama.cpp with CUDA support (omit -DGGML_CUDA=ON for CPU only)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

# Install Python requirements
pip install -r requirements.txt

# Convert a Hugging Face model to GGUF format
python3 convert_hf_to_gguf.py --outtype f16 \
  --outfile models/model-f16.gguf \
  /path/to/huggingface/model/

# Run inference
./build/bin/llama-cli -m ./models/model-f16.gguf \
  -ngl 99 -n 512 --ctx-size 4096 --temp 0.7 \
  -p "Explain the transformer architecture"
```

Key parameters for llama.cpp:

| Parameter | Description |
|-----------|-------------|
| `-ngl N` | Number of layers to offload to GPU (0 for CPU-only) |
| `-t N` | Number of CPU threads (use number of physical cores) |
| `-n N` | Maximum output tokens |
| `--ctx-size N` | Context window size (larger uses more RAM) |
| `--temp N` | Sampling temperature (0 = deterministic, higher = more creative) |

### Using vLLM for GPU Serving

```bash
# Activate virtual environment
source .venv/bin/activate

# Install vLLM
pip install vllm

# Serve a model with the vLLM OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --quantization awq \
  --dtype half
```

### Using torchtune for Fine-Tuning

torchtune is a PyTorch-native library for fine-tuning LLMs with composable building blocks.

```bash
# Activate virtual environment
source .venv/bin/activate

# Install torchtune
pip install torchtune

# Download a supported model from Hugging Face
tune download meta-llama/Llama-2-7b-hf \
  --output-dir /tmp/Llama-2-7b-hf \
  --hf-token <your_huggingface_token>

# Run a LoRA fine-tuning recipe with a config file
tune run lora_finetune_single_device \
  --config llama2/7B_lora_single_device
```

### Using Unsloth for Fast Fine-Tuning

The model setup code below is also available as the standalone script `fast_finetune.py`.
Run it with `python fast_finetune.py` after installing Unsloth and activating the virtual environment.
See the [Practical Example Scripts](#practical-example-scripts) section for setup instructions.

```bash
# Activate virtual environment
source .venv/bin/activate

# Install Unsloth (installs appropriate version for your CUDA version)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Example script: fast_finetune.py
```

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None  # Auto-detect
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                   # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

### Using DeepSpeed for Multi-GPU Training

DeepSpeed is a Microsoft library that reduces GPU memory requirements through the ZeRO (Zero Redundancy Optimizer) algorithm, which partitions optimizer states, gradients, and parameters across GPUs.

```bash
# Activate virtual environment
source .venv/bin/activate

# Install DeepSpeed
pip install deepspeed

# Run training with DeepSpeed ZeRO Stage 2
deepspeed --num_gpus=2 train.py \
  --deepspeed ds_config.json \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --output_dir ./output
```

Example `ds_config.json`:

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  },
  "fp16": {
    "enabled": true
  },
  "train_batch_size": 8
}
```

---

## Optimization on Microsoft Azure (Cloud)

Microsoft Azure provides an ecosystem for both training and inference optimization of LLMs.

### Azure Kubernetes Service (AKS) with KAITO

The Kubernetes AI Toolchain Operator (KAITO) automates LLM deployments in Kubernetes clusters on AKS. It:

- Automatically provisions right-sized GPU nodes
- Sets up inference servers as endpoint services
- Supports fine-tuning with LoRA via the KAITO Tuning Workspace API
- Allows storing and distributing LoRA adapter layers as container images

Setup steps:

1. Create an AKS cluster with GPU node pools.
2. Enable the AI toolchain operator add-on: `az aks update --enable-ai-toolchain-operator`.
3. Apply a KAITO workspace manifest specifying the model and compute requirements.
4. For fine-tuning: store training data as container images and define a tuning workspace.

### Azure Container Apps for LoRA Fine-Tuning

The Foundry Toolkit extension for VS Code enables fine-tuning models like Phi Silica using LoRA on Azure Container Apps:

1. Install the Foundry Toolkit extension in VS Code.
2. Navigate to **Tools > Fine-tuning**.
3. Select `microsoft/phi-silica` from the Model Catalog.
4. Upload `train.json` and `test.json` datasets.
5. Select the GPU workload profile (recommended: A100 GPUs, `Consumption-GPU-NC24-A100`).
6. Submit the fine-tuning job (typically 45-60 minutes).
7. Download the trained LoRA adapter (`.safetensors` file).

### Azure Machine Learning for Production Workflows

Azure Machine Learning (Azure ML) provides:

- **Managed compute clusters**: On-demand GPU/CPU clusters for training and inference
- **Model registry**: Versioned storage for model weights and adapters
- **Managed online endpoints**: Scalable, low-latency inference endpoints
- **Responsible AI dashboard**: Tools to monitor fairness, reliability, and interpretability

The job submission code below is also available as the standalone script `azure_ml_submit.py`.
Run it with `python azure_ml_submit.py` after setting the required environment variables.
See the [Practical Example Scripts](#practical-example-scripts) section for setup instructions.

```bash
# Install Azure ML SDK
pip install azure-ai-ml azure-identity

# Example: Submit a fine-tuning job to Azure ML
```

```python
from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="<subscription_id>",
    resource_group_name="<resource_group>",
    workspace_name="<workspace_name>",
)

job = command(
    code="./src",
    command="python fine_tune.py --model_name meta-llama/Llama-2-7b-hf --output_dir ./outputs",
    environment="azureml:AzureML-PyTorch-2.0-cuda11.8:1",
    compute="gpu-cluster",
    display_name="llm-lora-finetuning",
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
```

### Azure OpenAI Service Fine-Tuning

For models deployed through Azure OpenAI Service, fine-tuning is available via the Azure portal or REST API:

```bash
# Upload training data
curl -X POST "https://<endpoint>.openai.azure.com/openai/files?api-version=2024-02-01" \
  -H "api-key: <api-key>" \
  -F "purpose=fine-tune" \
  -F "file=@train.jsonl"

# Create a fine-tuning job
curl -X POST "https://<endpoint>.openai.azure.com/openai/fine_tuning/jobs?api-version=2024-02-01" \
  -H "api-key: <api-key>" \
  -H "Content-Type: application/json" \
  -d '{"training_file": "<file-id>", "model": "gpt-4o-mini"}'
```

---

## Microsoft Olive Optimization Workflow

Microsoft Olive is an end-to-end model optimization toolkit that takes a PyTorch or Hugging Face model and produces an optimized ONNX model ready for deployment on CPU, GPU, or NPU targets.

**Workflow:**

```
Input Model (PyTorch / Hugging Face)
         |
         v
  Microsoft Olive Pipeline
    ├── Graph Optimization (ONNX Runtime)
    ├── Quantization (INT4, INT8, FP8)
    └── Target-Specific Compilation (DirectML, QNN, CUDA)
         |
         v
  Optimized ONNX Model
  (for GPU, CPU, or NPU deployment)
```

**Setup:**

```bash
# Activate virtual environment
source .venv/bin/activate

# Install Microsoft Olive
pip install olive-ai[gpu]

# Install ONNX Runtime
pip install onnxruntime-gpu
```

**Example Olive configuration (`olive_config.json`):**

```json
{
  "input_model": {
    "type": "HfModel",
    "model_path": "microsoft/Phi-3-mini-4k-instruct"
  },
  "passes": {
    "conversion": { "type": "OnnxConversion" },
    "quantization": {
      "type": "OnnxMatMul4Quantizer",
      "precision": "int4"
    }
  },
  "output_dir": "./optimized_model"
}
```

```bash
# Run the Olive optimization pipeline
olive run --config olive_config.json
```

**Supported targets:**

- **GPU (CUDA)**: Optimized ONNX models with INT4/INT8 via ONNX Runtime
- **CPU**: Quantized models for CPU-only inference (useful for edge deployment)
- **NPU (DirectML / QNN)**: Optimized for Qualcomm or Intel NPU acceleration

---

## RAG Optimization with Azure AI Search

For production RAG pipelines, Microsoft Azure AI Search provides multi-vector retrieval and semantic ranking to improve grounding quality and reduce hallucinations.

The example code below is also available as the standalone script `azure_rag_search.py`.
Run it with `python azure_rag_search.py` after setting the required environment variables.
See the [Practical Example Scripts](#practical-example-scripts) section for setup instructions.

**Architecture:**

```
User Query
    |
    v
Azure AI Search (multi-vector retrieval)
    |  -- Semantic ranking of results
    v
Top-k Relevant Documents
    |
    v
Augmented Prompt (document + question)
    |
    v
LLM (Azure OpenAI / local model)
    |
    v
Grounded Response
```

**Setup:**

```bash
# Activate virtual environment
source .venv/bin/activate

# Install Azure SDK
pip install azure-search-documents azure-identity openai
```

**Example:**

```python
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

search_client = SearchClient(
    endpoint="https://<search-service>.search.windows.net",
    index_name="<index-name>",
    credential=AzureKeyCredential("<api-key>")
)

openai_client = AzureOpenAI(
    api_key="<azure-openai-key>",
    api_version="2024-02-01",
    azure_endpoint="https://<endpoint>.openai.azure.com/"
)

def rag_query(user_question: str) -> str:
    # Retrieve relevant context from Azure AI Search
    results = search_client.search(
        search_text=user_question,
        top=3,
        query_type="semantic",
        semantic_configuration_name="default"
    )
    context = "\n".join(r["content"] for r in results)

    # Send augmented prompt to the LLM
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer questions based on the provided context only."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_question}"}
        ]
    )
    return response.choices[0].message.content

answer = rag_query("What optimization techniques reduce VRAM usage?")
print(answer)
```

---

## Optimization on AWS and Google Cloud

### AWS: Running SLMs with llama.cpp on Local Zones and Outposts

AWS supports on-premises and edge deployment of small language models using llama.cpp on GPU-enabled EC2 instances (e.g., `g4dn.12xlarge`). The GGUF model format is used for optimized inference.

Key optimization parameters for llama.cpp on AWS:

- `-ngl 99`: Offload all layers to GPU
- `--ctx-size 8192`: Set context window (larger contexts use more RAM)
- `-sm row`: Row-based GPU splitting for multi-GPU setups
- `--temp 0`: Deterministic output for production use cases

For scalable deployments, an Application Load Balancer (ALB) fronts multiple inference instances. A RabbitMQ message queue can handle request bursts exceeding SLM capacity.

### Google Cloud: Fine-Tuning with Vertex AI

Google Cloud's Vertex AI provides a fully managed platform for fine-tuning foundation models. The workflow follows four steps:

1. **Data preparation**: Collect and split data into train/validation/test sets in the required format.
2. **Choose an approach**: Full fine-tuning (all parameters updated) or PEFT (LoRA/QLoRA, which freezes the base model and adds small trainable layers).
3. **Training**: Configure hyperparameters (learning rate, batch size, epochs) and monitor with validation metrics.
4. **Evaluation and deployment**: Evaluate on the test set and deploy to a Vertex AI endpoint.

Google Cloud hardware for fine-tuning includes TPUs (Tensor Processing Units), which are custom AI accelerators for large-scale training.

**Fine-tuning vs. RAG decision matrix:**

| Criterion | Fine-Tuning | RAG |
|-----------|------------|-----|
| Task-specific accuracy | High | Moderate |
| Domain-specific language | Excellent | Good |
| Up-to-date knowledge | Requires retraining | Dynamic |
| Compute cost | High | Low |
| Data required | Thousands of examples | None for the model |
| Hallucination risk | Lower after tuning | Low with good retrieval |

---

## Summary: Choosing the Right Approach

| Scenario | Recommended Approach |
|----------|---------------------|
| Pre-training from scratch | AdamW optimizer with BFloat16 mixed precision |
| Efficient fine-tuning on consumer GPU | QLoRA (4-bit base + LoRA adapters) |
| High-throughput inference on NVIDIA GPU | vLLM with PagedAttention, GQA-based models |
| Deploy on NVIDIA GPU cloud | TensorRT-LLM or vLLM |
| Efficient fine-tuning at scale | Hugging Face PEFT or NVIDIA NeMo |
| Training on AWS | SageMaker SDK with distributed training |
| Edge / on-premises CPU inference | llama.cpp with GGUF quantized models or Llamafile |
| Optimization for Windows NPU | Microsoft Olive with DirectML or QNN target |
| Domain-specific accuracy without retraining | RAG with Azure AI Search or FAISS |
| Leveraging frozen models with prompt context | Chain-of-Thought (CoT) or few-shot prompting |
| Optimizing prompts automatically | OPRO (Optimization by PROmpting) |
| Multi-GPU memory-efficient training | DeepSpeed ZeRO |
| Fast LLaMA/Mistral fine-tuning | Unsloth |

---

## References

1. Microsoft Azure. "What are large language models (LLMs)?"
   https://azure.microsoft.com/en-us/resources/cloud-computing-dictionary/what-are-large-language-models-llms

2. Microsoft Learn. "Fine-tuning language models for AI and machine learning workflows." Azure Kubernetes Service documentation.
   https://learn.microsoft.com/en-us/azure/aks/concepts-fine-tune-language-models

3. Microsoft Learn. "Small and large language models." Azure Kubernetes Service documentation.
   https://learn.microsoft.com/en-us/azure/aks/concepts-ai-ml-language-models

4. Microsoft Learn. "LoRA Fine-Tuning for Phi Silica." Windows AI documentation.
   https://learn.microsoft.com/en-us/windows/ai/apis/phi-silica-lora?tabs=csharp0

5. PyTorch Foundation. "torchtune: Easily fine-tune LLMs using PyTorch." PyTorch Blog, April 2024.
   https://pytorch.org/blog/torchtune-fine-tune-llms/

6. Lasiuk, Z. "Optimize LLMs for Efficiency and Sustainability." PyTorch Blog, February 2025.
   https://pytorch.org/blog/optimize-llms/

7. McEvilly, C., Galves, F., Ben Baruch, G. "Running and optimizing small language models on-premises and at the edge." AWS Compute Blog, June 2025.
   https://aws.amazon.com/blogs/compute/running-and-optimizing-small-language-models-on-premises-and-at-the-edge/

8. Google Cloud. "Fine-tuning LLMs and AI models."
   https://cloud.google.com/use-cases/fine-tuning-ai-models

9. Google Cloud. "Best practices for optimizing large language model inference with GPUs on Google Kubernetes Engine (GKE)."
   https://docs.cloud.google.com/kubernetes-engine/docs/best-practices/machine-learning/inference/llm-optimization

10. Unsloth. "Unsloth: Fast LLM Fine-Tuning." GitHub repository.
    https://github.com/unslothai/unsloth

11. DeepSpeed AI. "DeepSpeed: Extreme-scale model training for everyone." GitHub repository.
    https://github.com/deepspeedai/DeepSpeed

12. Vaswani, A., et al. "Attention is All You Need." NeurIPS, 2017.
    https://arxiv.org/abs/1706.03762

13. Hu, E., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR, 2022.
    https://arxiv.org/abs/2106.09685

14. Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS, 2023.
    https://arxiv.org/abs/2305.14314

15. Frankle, J., Carlin, M. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." ICLR, 2019.
    https://arxiv.org/abs/1803.03635

16. Hinton, G., Vinyals, O., Dean, J. "Distilling the Knowledge in a Neural Network." NIPS Deep Learning Workshop, 2014.
    https://arxiv.org/abs/1503.02531
