"""
Instruction-following chatbot built on the pretrained GPT model.

The chatbot uses a structured prompt template to simulate conversational
behavior. No instruction fine-tuning or RLHF is performed; instead, the
pretrained model completes a realistic-looking conversation transcript.
This is the "transcript hack" technique: a language model that has seen
conversational text during pretraining learns to continue dialogues.

For noticeably better responses, pretrain on a larger and more diverse corpus
or provide a corpus that contains conversation examples.

Usage:
    python chatbot.py
    python chatbot.py --temperature 0.9 --top-k 50 --max-tokens 300
"""

import argparse
import os
import sys

import torch

from model import GPTModel
from tokenizer import CharTokenizer

CHECKPOINT_PATH = os.path.join("checkpoints", "best_model.pt")

# Prompt template used to structure the conversation
SYSTEM_HEADER = "### System: You are a helpful AI assistant.\n\n"
HUMAN_PREFIX = "### Human: "
ASSISTANT_PREFIX = "\n### Assistant: "


def build_prompt(history: list[tuple[str, str]], new_user_input: str) -> str:
    """Construct a prompt string from conversation history and a new user message."""
    prompt = SYSTEM_HEADER
    for user_msg, assistant_msg in history:
        prompt += HUMAN_PREFIX + user_msg + ASSISTANT_PREFIX + assistant_msg + "\n\n"
    prompt += HUMAN_PREFIX + new_user_input + ASSISTANT_PREFIX
    return prompt


def load_model(checkpoint_path: str, device: str) -> tuple[GPTModel, CharTokenizer]:
    """Load the pretrained model and tokenizer from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run train.py first to generate a pretrained model.")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    chars = checkpoint["tokenizer_chars"]

    model = GPTModel(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.chars = chars
    tokenizer.vocab_size = len(chars)
    tokenizer.char_to_idx = {ch: i for i, ch in enumerate(chars)}
    tokenizer.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    return model, tokenizer


def generate_response(
    model: GPTModel,
    tokenizer: CharTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    """Generate the model's continuation of the prompt and extract the response."""
    encoded = tokenizer.encode(prompt)
    if not encoded:
        return "(Could not encode the prompt with the current vocabulary.)"

    context = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    generated = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    full_text = tokenizer.decode(generated[0].tolist())

    # Extract only the text that follows the last assistant prefix
    if ASSISTANT_PREFIX in full_text:
        # Take everything after the final occurrence of the assistant marker
        response = full_text.rsplit(ASSISTANT_PREFIX, 1)[-1]
        # Truncate at the next human turn if the model starts one
        if HUMAN_PREFIX in response:
            response = response.split(HUMAN_PREFIX)[0]
        return response.strip()

    # Fallback: return the newly generated portion only
    return full_text[len(prompt) :].strip()


def chat(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    model, tokenizer = load_model(args.checkpoint, device)
    print(f"Model loaded. Vocabulary size: {tokenizer.vocab_size}")
    print(
        "\nChatbot ready. Type a message and press Enter. "
        "Type 'exit' or 'quit' to stop.\n"
    )
    print("-" * 60)

    history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye.")
            break

        # Build the full prompt including conversation history
        prompt = build_prompt(history, user_input)

        response = generate_response(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        print(f"Assistant: {response}\n")

        # Keep a short conversation history (last 3 turns) to manage context length
        history.append((user_input, response))
        if len(history) > 3:
            history.pop(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chatbot using a pretrained GPT model.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH, help="Path to model checkpoint")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate per response")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.1-1.5)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (0 to disable)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.top_k == 0:
        args.top_k = None
    chat(args)
