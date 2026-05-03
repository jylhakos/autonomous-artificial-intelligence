"""
prompt_optimization.py
======================
Demonstrates few-shot prompt optimization (in-context learning) using a
small GPT-2 model via the Hugging Face Transformers pipeline API.

README section: "Example Usage: Few-Shot Prompt Optimization"

Concept covered:
    Few-shot prompting improves model accuracy without any weight updates.
    By embedding input-output examples directly in the prompt (the "few
    shots"), the model learns the desired output format and task at inference
    time.  This is also the basis of Chain-of-Thought (CoT) prompting, where
    intermediate reasoning steps are provided as examples.

    This technique is described in the README under:
    - "Inference Optimization Techniques" (prompt-level optimization)
    - "Summary: Choosing the Right Approach" (leveraging frozen models)

    OPRO (Optimization by PROmpting) is a related technique where the LLM
    itself is used as an optimizer to discover better prompts based on task
    performance metrics.

Requirements:
    - Virtual environment activated (see README "Python Virtual Environment Setup")

    pip install transformers torch
    # or:
    pip install -r requirements.txt

Usage:
    # 1. Activate the virtual environment
    source .venv/bin/activate

    # 2. Run the script
    python prompt_optimization.py
"""

from transformers import pipeline

# ---------------------------------------------------------------------------
# Load a small model (no GPU required for GPT-2)
# ---------------------------------------------------------------------------
print("Loading GPT-2 text-generation pipeline ...")
generator = pipeline("text-generation", model="gpt2", max_new_tokens=10)

# ---------------------------------------------------------------------------
# Build a few-shot prompt
# The model sees three labelled examples before the unlabelled test sentence.
# This guides the model to follow the "Sentiment:" format without training.
# ---------------------------------------------------------------------------
few_shot_prompt = """Classify the sentiment of the following sentences.

Sentence: "This product is terrible."
Sentiment: Negative

Sentence: "Absolutely loved it!"
Sentiment: Positive

Sentence: "It was okay, nothing special."
Sentiment: Neutral

Sentence: "Okay."
Sentiment:"""

print("\nFew-shot prompt:")
print(few_shot_prompt)
print("\nModel output:")

output = generator(few_shot_prompt, do_sample=False)
print(output[0]["generated_text"])

# ---------------------------------------------------------------------------
# Demonstrate that without the few-shot examples the output is unstructured
# ---------------------------------------------------------------------------
zero_shot_prompt = 'Classify the sentiment of: "Okay."'
print("\n--- Zero-shot (no examples) for comparison ---")
print(f"Prompt: {zero_shot_prompt}")
zero_output = generator(zero_shot_prompt, do_sample=False)
print(zero_output[0]["generated_text"])
