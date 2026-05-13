"""
self_instruct_generator.py

Synthetic data creation using the Self-Instruct technique with Ollama.

This script implements the four-step Self-Instruct pipeline:
  1. Instruction Generation  - use an LLM to generate new instructions from seed tasks
  2. Input/Output Generation - generate input-output pairs for each instruction
  3. Filtering               - remove low-quality or near-duplicate instructions (ROUGE-L)
  4. Pool Update             - add high-quality instructions back into the seed pool

References:
  - Self-Instruct paper: https://arxiv.org/abs/2212.10560
  - Official repository:  https://github.com/yizhongw/self-instruct
  - HuggingFace overview: https://huggingface.co/blog/davanstrien/self-instruct
  - Alpaca (Stanford):    https://crfm.stanford.edu/2023/03/13/alpaca.html

Requirements (installed inside .venv):
  ollama, rouge-score, tqdm

Usage:
  python scripts/self_instruct_generator.py --model llama3 --num-instructions 20 --output output/dataset.jsonl
"""

import argparse
import json
import re
import os
from pathlib import Path

try:
    import ollama
except ImportError:
    raise ImportError(
        "The 'ollama' package is not installed. "
        "Activate the virtual environment and run: pip install ollama"
    )

try:
    from rouge_score import rouge_scorer
except ImportError:
    raise ImportError(
        "The 'rouge-score' package is not installed. "
        "Activate the virtual environment and run: pip install rouge-score"
    )

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError(
        "The 'tqdm' package is not installed. "
        "Activate the virtual environment and run: pip install tqdm"
    )

from seed_tasks import SEED_TASKS, get_seed_instructions

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "llama3"
DEFAULT_NUM_INSTRUCTIONS = 20
DEFAULT_OUTPUT = "output/dataset.jsonl"
ROUGE_SIMILARITY_THRESHOLD = 0.7   # discard if ROUGE-L F1 >= this value vs any existing
NUM_SEED_EXAMPLES_IN_PROMPT = 8     # how many seed examples to include in each prompt


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_instruction_prompt(existing_instructions: list) -> str:
    """
    Build a prompt that shows existing instructions and asks the model
    to generate new, diverse ones.  This mirrors the Self-Instruct approach.
    """
    sample = existing_instructions[-NUM_SEED_EXAMPLES_IN_PROMPT:]
    lines = "\n".join(f"{i + 1}. {instr}" for i, instr in enumerate(sample))
    prompt = (
        "You are an expert task generator for a language model training dataset.\n"
        "Study the following example tasks:\n\n"
        f"{lines}\n\n"
        "Generate 5 new, diverse, and creative task instructions for a language model. "
        "Make the instructions varied in topic, length, and complexity. "
        "Do NOT repeat any of the examples above.\n\n"
        "Format your response exactly as:\n"
        "Instruction: <task instruction>\n"
        "Instruction: <task instruction>\n"
        "...\n"
    )
    return prompt


def build_instance_prompt(instruction: str) -> str:
    """
    Build a prompt that asks the model to generate the ideal output
    for a given instruction.
    """
    return (
        f"Complete the following task and provide the ideal response.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Output:"
    )


# ---------------------------------------------------------------------------
# Step 1 – Instruction Generation
# ---------------------------------------------------------------------------

def generate_instructions(model: str, existing_instructions: list, batch_size: int = 5) -> list:
    """
    Ask the model to generate new instructions based on existing ones.
    Returns a list of new instruction strings.
    """
    prompt = build_instruction_prompt(existing_instructions)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        raw_text = response["response"]
    except Exception as exc:
        print(f"[ERROR] Ollama generation failed: {exc}")
        return []

    # Parse lines that start with "Instruction:"
    new_instructions = []
    for line in raw_text.splitlines():
        line = line.strip()
        if line.lower().startswith("instruction:"):
            instruction_text = line[len("instruction:"):].strip()
            if instruction_text:
                new_instructions.append(instruction_text)

    return new_instructions


# ---------------------------------------------------------------------------
# Step 2 – Input/Output Generation
# ---------------------------------------------------------------------------

def generate_output(model: str, instruction: str) -> str:
    """
    Ask the model to produce the ideal output for a given instruction.
    Returns the generated output string.
    """
    prompt = build_instance_prompt(instruction)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        return response["response"].strip()
    except Exception as exc:
        print(f"[ERROR] Output generation failed for instruction: {instruction[:60]}... | {exc}")
        return ""


# ---------------------------------------------------------------------------
# Step 3 – Filtering (ROUGE-L similarity)
# ---------------------------------------------------------------------------

def is_too_similar(candidate: str, existing: list, threshold: float) -> bool:
    """
    Returns True if the candidate instruction is too similar to any existing
    instruction based on ROUGE-L F1 score.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for existing_instr in existing:
        scores = scorer.score(candidate.lower(), existing_instr.lower())
        if scores["rougeL"].fmeasure >= threshold:
            return True
    return False


def passes_quality_checks(instruction: str, output: str) -> bool:
    """
    Apply basic heuristic quality filters:
      - instruction must be at least 10 characters
      - output must not be empty
      - instruction and output must not be identical
      - instruction must contain at least one verb-like token
    """
    if len(instruction) < 10:
        return False
    if not output:
        return False
    if instruction.strip().lower() == output.strip().lower():
        return False
    # Reject if the model output looks like an error or refusal
    refusal_patterns = ["i cannot", "i'm sorry", "as an ai", "i don't know"]
    if any(pat in output.lower() for pat in refusal_patterns):
        return False
    return True


# ---------------------------------------------------------------------------
# Step 4 – Pool Update and JSONL persistence
# ---------------------------------------------------------------------------

def save_entry(entry: dict, output_path: str) -> None:
    """Append a single dataset entry to a JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(model: str, num_instructions: int, output_path: str) -> None:
    """
    Execute the full Self-Instruct pipeline.

    How do we generate instructions without writing them all by hand?
    ----------------------------------------------------------------
    The answer: we start with a small set of human-written 'seed' tasks
    and iteratively prompt an LLM to generate new instructions that are
    diverse (filtered by ROUGE-L) and complete (have a non-empty output).
    Each accepted instruction is added back into the instruction pool so
    subsequent generation rounds produce increasingly complex tasks.
    """
    print(f"\nSelf-Instruct Pipeline")
    print(f"  Model            : {model}")
    print(f"  Target count     : {num_instructions}")
    print(f"  Output file      : {output_path}")
    print(f"  ROUGE-L threshold: {ROUGE_SIMILARITY_THRESHOLD}\n")

    # Initialise the instruction pool with seed tasks
    instruction_pool: list = get_seed_instructions()
    generated_entries: list = []
    accepted = 0

    progress = tqdm(total=num_instructions, desc="Generating instructions", unit="instr")

    while accepted < num_instructions:
        # --- Step 1: generate a batch of candidate instructions ---
        candidates = generate_instructions(model, instruction_pool)

        for candidate in candidates:
            if accepted >= num_instructions:
                break

            # --- Step 3: filter for similarity ---
            if is_too_similar(candidate, instruction_pool, ROUGE_SIMILARITY_THRESHOLD):
                tqdm.write(f"[SKIP – too similar] {candidate[:70]}...")
                continue

            # --- Step 2: generate output for the instruction ---
            output_text = generate_output(model, candidate)

            # --- Step 3: quality checks ---
            if not passes_quality_checks(candidate, output_text):
                tqdm.write(f"[SKIP – quality]    {candidate[:70]}...")
                continue

            # --- Step 4: accept and update pool ---
            entry = {
                "id": f"generated_{accepted + 1:04d}",
                "instruction": candidate,
                "input": "",
                "output": output_text,
            }
            save_entry(entry, output_path)
            generated_entries.append(entry)
            instruction_pool.append(candidate)   # grow the pool
            accepted += 1
            progress.update(1)
            tqdm.write(f"[ACCEPTED {accepted:3d}]     {candidate[:70]}...")

    progress.close()
    print(f"\nDone. {accepted} instructions saved to '{output_path}'.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic instruction-tuning dataset using "
            "the Self-Instruct technique with a local Ollama model."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use for generation (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num-instructions",
        type=int,
        default=DEFAULT_NUM_INSTRUCTIONS,
        help=f"Number of instructions to generate (default: {DEFAULT_NUM_INSTRUCTIONS})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to the output JSONL file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        model=args.model,
        num_instructions=args.num_instructions,
        output_path=args.output,
    )
