"""
Train Your Own Reasoning Model with Unsloth (GRPO)
===================================================
Implements Group Relative Policy Optimization (GRPO) to convert a standard
instruction-tuned LLM into a reasoning model that generates explicit
Chain-of-Thought traces before producing a final answer.

Method:   DeepSeek-R1-Zero approach scaled to consumer hardware via Unsloth.
Dataset:  openai/gsm8k  (grade-school math, multi-step arithmetic)
Hardware: 7 GB VRAM minimum (Qwen2.5-1.5B via QLoRA)
          15 GB VRAM for models up to 15B (Llama 3.1-8B, Phi-4-14B)

References
----------
- Train your own R1 reasoning model (Unsloth):
    https://unsloth.ai/blog/r1-reasoning
- Practical Exercise: GRPO with Unsloth (HuggingFace LLM Course):
    https://huggingface.co/learn/llm-course/en/chapter12/6
- Reinforcement Learning (RL) Guide (Unsloth Docs):
    https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
- Long-context GRPO:
    https://unsloth.ai/blog/grpo
- DeepSeek-R1 technical report:
    https://arxiv.org/abs/2501.12948
- GSM8K dataset:
    https://huggingface.co/datasets/openai/gsm8k
- OpenMathReasoning-mini dataset (alternative, better reasoning quality):
    https://huggingface.co/datasets/nvidia/OpenMathReasoning-mini
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit these values or pass them as CLI arguments
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "google/gemma-3-1b-it"
# Alternative models (uncomment one):
#   "unsloth/Llama-3.1-8B-Instruct"  — 15 GB VRAM
#   "unsloth/Qwen2.5-1.5B-Instruct"  —  7 GB VRAM (minimum)
#   "unsloth/Phi-4-14B-Instruct"     — 15 GB VRAM

DEFAULT_DATASET = "gsm8k"
# Alternatives:
#   "openmath"  -> nvidia/OpenMathReasoning-mini (better reasoning quality, SFT path)

MAX_SEQ_LENGTH = 1024    # Increase for longer reasoning traces (e.g. 2048, 4096)
LORA_RANK = 32           # Larger rank = more expressive, but slower. Try 16 or 64.
MAX_STEPS = 250          # Increase to 1000+ for production quality (12+ hours)
NUM_GENERATIONS = 6      # Completions per prompt. Decrease if out of memory.
OUTPUT_DIR = "outputs"

# ---------------------------------------------------------------------------
# Structured output format
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_xml_answer(text: str) -> str:
    """Extract the content inside <answer>...</answer> tags."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    """Extract the final answer after '####' delimiter (GSM8K format)."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def load_gsm8k(split: str = "train"):
    """
    Load and format the GSM8K dataset for GRPO training.

    GSM8K: Grade-school math word problems with step-by-step solutions.
    Each example has a 'question' and an 'answer' field where the final
    numeric answer follows a '####' delimiter.

    Dataset: https://huggingface.co/datasets/openai/gsm8k
    """
    from datasets import load_dataset, Dataset  # noqa: PLC0415

    print(f"Loading GSM8K dataset (split={split}) ...")
    data = load_dataset("openai/gsm8k", "main")[split]

    def _format(example: dict) -> dict:
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": extract_hash_answer(example["answer"]),
        }

    data = data.map(_format)
    print(f"  Loaded {len(data)} examples.")
    print(f"  Sample question : {data[0]['prompt'][1]['content'][:80]}...")
    print(f"  Sample answer   : {data[0]['answer']}")
    return data


def load_openmath(split: str = "train"):
    """
    Load and format the nvidia/OpenMathReasoning-mini dataset.

    This dataset contains math problems paired with 'generated_solution'
    reasoning traces — ideal for both SFT and GRPO paths.

    Dataset: https://huggingface.co/datasets/nvidia/OpenMathReasoning-mini
    """
    from datasets import load_dataset  # noqa: PLC0415

    print("Loading nvidia/OpenMathReasoning-mini dataset ...")
    data = load_dataset("nvidia/OpenMathReasoning-mini", split=split)

    def _format(example: dict) -> dict:
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
            # The 'expected_answer' column holds the final numeric/symbolic answer.
            "answer": str(example.get("expected_answer", "")),
        }

    data = data.map(_format)
    print(f"  Loaded {len(data)} examples.")
    return data


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def correctness_reward_func(
    prompts: list,
    completions: list,
    answer: list[str],
    **kwargs,
) -> list[float]:
    """
    Reward +2.0 when the extracted answer matches the ground-truth answer.
    This is the most important signal — it drives the model toward correct answers.
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]


def int_reward_func(completions: list, **kwargs) -> list[float]:
    """
    Reward +0.5 when the extracted answer is a pure integer.
    Useful for math datasets where the answer is always numeric.
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted]


def strict_format_reward_func(completions: list, **kwargs) -> list[float]:
    """
    Reward +0.5 for exact compliance with the XML structure:
      <reasoning>\\n...\\n</reasoning>\\n<answer>\\n...\\n</answer>\\n
    """
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]


def soft_format_reward_func(completions: list, **kwargs) -> list[float]:
    """
    Reward +0.5 for a relaxed XML structure (whitespace-tolerant).
    Used alongside strict_format to provide a gradient signal even when
    the model is still learning exact formatting.
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if m else 0.0 for m in matches]


def count_xml(text: str) -> float:
    """
    Fine-grained XML tag reward:
      +0.125 for each correct tag placement (4 tags = 0.5 max)
       -0.001 per character of trailing content after </answer>
    Penalising trailing content prevents the model from appending
    extra text after the closing answer tag.
    """
    score = 0.0
    if text.count("<reasoning>\n") == 1:
        score += 0.125
    if text.count("\n</reasoning>\n") == 1:
        score += 0.125
    if text.count("\n<answer>\n") == 1:
        score += 0.125
        score -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        score += 0.125
        score -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return score


def xmlcount_reward_func(completions: list, **kwargs) -> list[float]:
    """Applies count_xml to each completion."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# All reward functions composed for the trainer
REWARD_FUNCS = [
    xmlcount_reward_func,        # Tag structure hygiene (up to +0.5)
    soft_format_reward_func,     # Lenient format (up to +0.5)
    strict_format_reward_func,   # Exact format (up to +0.5)
    int_reward_func,             # Numeric answer (up to +0.5)
    correctness_reward_func,     # Ground-truth match (up to +2.0)
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, lora_rank: int, max_seq_length: int):
    """
    Load a pre-trained model with Unsloth optimisations:
      - 4-bit quantisation (load_in_4bit) to minimise VRAM footprint.
      - fast_inference enables the vLLM backend for generation during GRPO.
      - LoRA adapters are applied to attention and MLP projection layers.
      - use_gradient_checkpointing="unsloth" enables long-context training.

    Unsloth reduces VRAM consumption by ~80 % compared to HuggingFace + FA2.
    """
    from unsloth import FastLanguageModel  # noqa: PLC0415

    print(f"Loading model: {model_name}")
    print(f"  max_seq_length={max_seq_length}, lora_rank={lora_rank}, load_in_4bit=True")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,           # 4-bit NF4 quantisation — halves VRAM vs fp16
        fast_inference=True,         # Use vLLM for fast token generation in GRPO
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,  # Reduce to 0.5 if OOM errors occur
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            # Attention projections
            "q_proj", "k_proj", "v_proj", "o_proj",
            # MLP projections (remove if OOM)
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",  # Enables long-context fine-tuning
        random_state=3407,
    )

    print("  Model and LoRA adapters loaded successfully.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_training_args(
    max_seq_length: int,
    max_steps: int,
    num_generations: int,
    output_dir: str,
) -> object:
    """
    Build GRPOConfig training hyperparameters.

    Key parameters:
      num_generations   — completions per prompt; more = better signal, more VRAM
      max_completion_length — budget for reasoning + answer tokens
      paged_adamw_8bit  — 8-bit paged optimiser reduces optimiser state VRAM
      max_grad_norm     — gradient clipping for stable RL training
    """
    from trl import GRPOConfig  # noqa: PLC0415

    max_prompt_length = 256

    return GRPOConfig(
        # Optimiser
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        max_grad_norm=0.1,
        # Batch and sequence sizes
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,   # Increase to 4 for smoother gradient estimates
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        # Training duration
        max_steps=max_steps,
        save_steps=max_steps,
        # Logging
        logging_steps=1,
        report_to="none",   # Replace with "wandb" or "tensorboard" for tracking
        output_dir=output_dir,
    )


def run_training(
    model,
    tokenizer,
    dataset,
    training_args,
) -> None:
    """
    Run GRPO training loop via TRL's GRPOTrainer.

    The GRPO cycle per step:
      1. Generate `num_generations` completions for each prompt in the batch.
      2. Score each completion with every reward function.
      3. Sum reward scores; compute group mean.
      4. Update model to favour completions scoring above the group mean.

    Training guidance:
      - Expect no reward improvement for the first 150-200 steps — this is normal.
      - Wait at least 300 steps before judging convergence.
      - For production quality, train for at least 12 hours (1000+ steps).
      - Models below 1.5B parameters may not generate thinking tokens reliably.
    """
    from trl import GRPOTrainer  # noqa: PLC0415

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=dataset,
    )

    print("\nStarting GRPO training ...")
    print(f"  Steps        : {training_args.max_steps}")
    print(f"  Generations  : {training_args.num_generations}")
    print(f"  Output dir   : {training_args.output_dir}")
    print("  Reward funcs : xmlcount, soft_format, strict_format, int, correctness")
    print("  Note: rewards typically do not increase until step ~300.\n")

    trainer.train()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Save and export
# ---------------------------------------------------------------------------

def save_model(model, tokenizer, output_dir: str) -> None:
    """
    Save the trained model in three formats:
      1. LoRA adapter weights only   — smallest, for continued fine-tuning
      2. Merged fp16 model           — full model for local vLLM inference
      3. GGUF (q4_k_m)              — quantised for Ollama / llama.cpp

    The GGUF file can be loaded into Ollama with a Modelfile:
      FROM ./outputs/model_merged_gguf/model-q4_k_m.gguf
      SYSTEM "..."
    """
    lora_path = str(Path(output_dir) / "grpo_saved_lora")
    merged_path = str(Path(output_dir) / "model_merged")
    gguf_path = str(Path(output_dir) / "model_merged_gguf")

    print(f"\nSaving LoRA adapter weights to: {lora_path}")
    model.save_lora(lora_path)

    print(f"Saving merged fp16 model to: {merged_path}")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    print(f"Saving GGUF (q4_k_m) to: {gguf_path}")
    model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")

    print("All model artefacts saved.")


def test_model(model, tokenizer, output_dir: str) -> None:
    """
    Run a quick inference test with the trained LoRA adapter.
    Demonstrates that the model produces structured <reasoning>/<answer> output.
    """
    from vllm import SamplingParams  # noqa: PLC0415

    lora_path = str(Path(output_dir) / "grpo_saved_lora")
    test_question = "A train travels at 60 mph for 3 hours. How far does it travel?"

    print(f"\nTest question: {test_question}")
    print("Generating response ...\n")

    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

    output = (
        model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=model.load_lora(lora_path),
        )[0]
        .outputs[0]
        .text
    )

    print(output)
    # Expected: <reasoning>Distance = speed x time = 60 x 3 = 180 miles</reasoning>
    #           <answer>180 miles</answer>


# ---------------------------------------------------------------------------
# Push to HuggingFace Hub (optional)
# ---------------------------------------------------------------------------

def push_to_hub(model, tokenizer, repo_id: str, hf_token: str) -> None:
    """
    Push the trained model to the HuggingFace Hub in three quantisation formats.
    The GGUF files can be loaded directly in Ollama or llama.cpp.

    Requires a HuggingFace account and write-access token:
      https://huggingface.co/settings/tokens
    """
    print(f"\nPushing model to HuggingFace Hub: {repo_id}")
    model.push_to_hub_gguf(
        repo_id,
        tokenizer,
        quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
        token=hf_token,
    )
    print("Push complete.")


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a reasoning model with GRPO via Unsloth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "openmath"],
        default=DEFAULT_DATASET,
        help=(
            "Training dataset. "
            "'gsm8k' = openai/gsm8k (GRPO path, question+answer only). "
            "'openmath' = nvidia/OpenMathReasoning-mini (includes reasoning traces)."
        ),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="Maximum sequence length (prompt + reasoning + answer).",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=LORA_RANK,
        help="LoRA rank (r). Higher = more expressive but slower.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="Total GRPO training steps. Use 1000+ for production quality.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=NUM_GENERATIONS,
        help="Completions generated per prompt per step. Decrease if OOM.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory to save LoRA weights, merged model, and GGUF files.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip post-training inference test.",
    )
    parser.add_argument(
        "--push-to-hub",
        metavar="USERNAME/REPO",
        default=None,
        help="Optional HuggingFace Hub repo to push GGUF exports (e.g. 'alice/my-reasoning-model').",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace write token (required when --push-to-hub is set).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- GPU check ---
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            print(
                "WARNING: No CUDA GPU detected. GRPO training on CPU is extremely slow\n"
                "and not recommended. Minimum: 7 GB VRAM (e.g. RTX 3060 / T4).\n"
            )
        else:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU  : {torch.cuda.get_device_name(0)}")
            print(f"VRAM : {vram_gb:.1f} GB")
            if vram_gb < 7:
                print(
                    f"WARNING: {vram_gb:.1f} GB VRAM is below the 7 GB minimum.\n"
                    "Consider using a smaller model (Qwen2.5-1.5B) or reducing\n"
                    "--lora-rank to 8 and --num-generations to 4.\n"
                )
    except ImportError:
        print("torch not found — install it before running training.")
        sys.exit(1)

    # --- Load dataset ---
    dataset_loaders = {"gsm8k": load_gsm8k, "openmath": load_openmath}
    dataset = dataset_loaders[args.dataset]()

    # --- Load model ---
    model, tokenizer = load_model(
        model_name=args.model,
        lora_rank=args.lora_rank,
        max_seq_length=args.max_seq_length,
    )

    # --- Build training config ---
    training_args = build_training_args(
        max_seq_length=args.max_seq_length,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        output_dir=args.output_dir,
    )

    # --- Train ---
    run_training(model, tokenizer, dataset, training_args)

    # --- Save artefacts ---
    save_model(model, tokenizer, args.output_dir)

    # --- Quick inference test ---
    if not args.skip_test:
        test_model(model, tokenizer, args.output_dir)

    # --- Optional Hub push ---
    if args.push_to_hub:
        if not args.hf_token:
            print("ERROR: --hf-token is required when --push-to-hub is set.")
            sys.exit(1)
        push_to_hub(model, tokenizer, args.push_to_hub, args.hf_token)

    print("\nDone. Model artefacts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
