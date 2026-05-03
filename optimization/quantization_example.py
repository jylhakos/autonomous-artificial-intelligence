"""
quantization_example.py
=======================
Demonstrates 4-bit Post-Training Quantization (PTQ) using the Hugging Face
Transformers library together with bitsandbytes.

README section: "Example Usage: 4-Bit Quantization with Hugging Face"

Concept covered:
    Quantization reduces the numerical precision of model weights and
    activations from FP16 (2 bytes per parameter) to INT4 (0.5 bytes per
    parameter).  For a 7B-parameter model this cuts VRAM from ~14 GB down to
    ~3.5 GB, enabling inference on consumer-grade GPUs (8–12 GB VRAM) with
    minimal accuracy loss.

    The NF4 (NormalFloat4) data type used here is the quantization format
    introduced by QLoRA (Dettmers et al., 2023).  Weights are stored in 4-bit
    but dequantized to FP16 on-the-fly for every matrix multiplication.

Requirements:
    - NVIDIA GPU with CUDA support (or CPU for very small models)
    - Virtual environment activated (see README "Python Virtual Environment Setup")

    pip install torch transformers bitsandbytes accelerate
    # or:
    pip install -r requirements.txt

Usage:
    # 1. Activate the virtual environment
    source .venv/bin/activate

    # 2. Run the script
    python quantization_example.py

Note:
    Llama-2-7b-hf is a gated model.  You need a Hugging Face account and to
    accept the model licence at https://huggingface.co/meta-llama/Llama-2-7b-hf
    before downloading it.  Authenticate with:
        huggingface-cli login
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
model_id = "meta-llama/Llama-2-7b-hf"

# ---------------------------------------------------------------------------
# Configure 4-bit quantization with NF4 format (QLoRA-style)
# ---------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # NormalFloat4 data type
    bnb_4bit_compute_dtype=torch.float16, # Compute in FP16 after dequantization
)

# ---------------------------------------------------------------------------
# Load tokenizer and quantized model
# ---------------------------------------------------------------------------
print(f"Loading tokenizer for {model_id} ...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Loading model in 4-bit precision ...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically distribute across available GPUs / CPU
)

# ---------------------------------------------------------------------------
# Report memory usage
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory allocated: {mem_gb:.2f} GB  (FP16 baseline would be ~14 GB)")

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
prompt = "Explain the concept of quantization in machine learning:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print(f"\nPrompt: {prompt}\n")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(output[0], skip_special_tokens=True))
