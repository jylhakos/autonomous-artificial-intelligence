"""
dataset_cleaner.py

Utilities for cleaning and transforming raw text datasets before
pre-training or fine-tuning a Large Language Model or a Machine Learning model.

Covers:
  - HTML tag removal
  - URL removal
  - Unicode normalisation to UTF-8
  - Whitespace normalisation
  - Duplicate detection (exact and near-duplicate via MinHash + Jaccard)
  - Perplexity-based quality filter stub (requires a language model)
  - JSONL dataset loading and saving

References:
  - AWS dataset preparation guide:
      https://aws.amazon.com/blogs/machine-learning/an-introduction-to-preparing-your-own-dataset-for-llm-training/
  - Fine-tuning overview (Google Cloud):
      https://cloud.google.com/use-cases/fine-tuning-ai-models
"""

import re
import json
import hashlib
import unicodedata
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_NOISY_CHARS_RE = re.compile(r"[^\w\s.,!?;:()\[\]{}\-\"\'`@#$%&*+=/<>\\|~^]")


def remove_html_tags(text: str) -> str:
    """Strip HTML markup tags from text."""
    return _HTML_TAG_RE.sub(" ", text)


def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs and bare www addresses."""
    return _URL_RE.sub(" ", text)


def normalize_unicode(text: str) -> str:
    """
    Normalise unicode to NFC form and encode/decode to UTF-8,
    dropping characters that cannot be represented.
    """
    text = unicodedata.normalize("NFC", text)
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def remove_noisy_characters(text: str) -> str:
    """Remove unusual non-ASCII symbols that add noise to training data."""
    return _NOISY_CHARS_RE.sub("", text)


def clean_text(text: str, remove_urls_flag: bool = True) -> str:
    """
    Full cleaning pipeline applied in order:
      1. Remove HTML tags
      2. Optionally remove URLs
      3. Normalise unicode
      4. Remove noisy characters
      5. Normalise whitespace
    """
    text = remove_html_tags(text)
    if remove_urls_flag:
        text = remove_urls(text)
    text = normalize_unicode(text)
    text = remove_noisy_characters(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Deduplication helpers (MinHash / Jaccard approximation)
# ---------------------------------------------------------------------------

def _shingles(text: str, k: int = 5) -> set:
    """
    Create a set of character-level k-shingles from text.
    Lower-cased and stripped for consistency.
    """
    text = text.lower().strip()
    return {text[i: i + k] for i in range(max(len(text) - k + 1, 1))}


def _minhash_signature(shingle_set: set, num_hashes: int = 128) -> list:
    """
    Compute a MinHash signature (list of minimum hash values)
    for an approximate Jaccard similarity measure.
    """
    signature = []
    shingle_list = sorted(shingle_set)
    for i in range(num_hashes):
        min_hash = float("inf")
        for shingle in shingle_list:
            h = int(hashlib.md5(f"{i}:{shingle}".encode()).hexdigest(), 16)
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return signature


def jaccard_from_minhash(sig_a: list, sig_b: list) -> float:
    """
    Estimate Jaccard similarity between two MinHash signatures.
    """
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must have the same length.")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def exact_hash(text: str) -> str:
    """Return the SHA-256 hash of a text string for exact deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ExactDeduplicator:
    """Track and remove exact duplicate text entries."""

    def __init__(self):
        self._seen: set = set()

    def is_duplicate(self, text: str) -> bool:
        h = exact_hash(text)
        if h in self._seen:
            return True
        self._seen.add(h)
        return False


# ---------------------------------------------------------------------------
# Quality filter stubs
# ---------------------------------------------------------------------------

def passes_length_filter(text: str, min_words: int = 5, max_words: int = 4096) -> bool:
    """
    Reject texts that are too short or unreasonably long.
    A simple but effective first-pass quality gate.
    """
    word_count = len(text.split())
    return min_words <= word_count <= max_words


def passes_alpha_ratio_filter(text: str, min_ratio: float = 0.6) -> bool:
    """
    Reject text where fewer than min_ratio of characters are alphabetic.
    Helps filter out garbled or symbol-heavy content.
    """
    if not text:
        return False
    alpha_chars = sum(1 for c in text if c.isalpha())
    return (alpha_chars / len(text)) >= min_ratio


def passes_repetition_filter(text: str, max_repetition_ratio: float = 0.3) -> bool:
    """
    Reject text with excessive repeated n-grams (a common signal of low quality).
    """
    words = text.lower().split()
    if len(words) < 10:
        return True
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    unique_ratio = len(set(bigrams)) / len(bigrams)
    return unique_ratio >= (1 - max_repetition_ratio)


def apply_quality_filters(text: str) -> bool:
    """
    Apply all quality heuristics.  Returns True if the text passes all filters.
    """
    return (
        passes_length_filter(text)
        and passes_alpha_ratio_filter(text)
        and passes_repetition_filter(text)
    )


# ---------------------------------------------------------------------------
# JSONL dataset I/O
# ---------------------------------------------------------------------------

def load_jsonl(filepath: str) -> Iterator[dict]:
    """Yield records from a JSONL file one line at a time."""
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl(records: list, filepath: str) -> None:
    """Write a list of dicts to a JSONL file, creating parent dirs if needed."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Dataset cleaning pipeline
# ---------------------------------------------------------------------------

def clean_dataset(
    input_path: str,
    output_path: str,
    text_fields: list = None,
    deduplicate: bool = True,
) -> dict:
    """
    Load a JSONL dataset, clean all specified text fields, apply quality
    filters, and optionally deduplicate.

    Parameters
    ----------
    input_path   : path to the input JSONL file
    output_path  : path to write the cleaned JSONL file
    text_fields  : list of field names to clean (default: ['instruction', 'input', 'output'])
    deduplicate  : whether to remove exact duplicate instructions

    Returns a summary dict with counts.
    """
    if text_fields is None:
        text_fields = ["instruction", "input", "output"]

    deduplicator = ExactDeduplicator()
    cleaned_records = []
    total = 0
    skipped_quality = 0
    skipped_duplicate = 0

    for record in load_jsonl(input_path):
        total += 1

        # Clean text fields
        for field in text_fields:
            if field in record and isinstance(record[field], str):
                record[field] = clean_text(record[field])

        # Quality check on the primary instruction field
        primary_text = record.get("instruction", record.get("text", ""))
        if not apply_quality_filters(primary_text):
            skipped_quality += 1
            continue

        # Deduplication
        if deduplicate and deduplicator.is_duplicate(primary_text):
            skipped_duplicate += 1
            continue

        cleaned_records.append(record)

    save_jsonl(cleaned_records, output_path)

    summary = {
        "total_input": total,
        "accepted": len(cleaned_records),
        "skipped_quality": skipped_quality,
        "skipped_duplicate": skipped_duplicate,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean and deduplicate a JSONL instruction dataset."
    )
    parser.add_argument("--input", required=True, help="Path to raw JSONL file")
    parser.add_argument("--output", required=True, help="Path to write cleaned JSONL file")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["instruction", "input", "output"],
        help="Text fields to clean (default: instruction input output)",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable exact deduplication",
    )
    args = parser.parse_args()

    summary = clean_dataset(
        input_path=args.input,
        output_path=args.output,
        text_fields=args.fields,
        deduplicate=not args.no_dedup,
    )

    print("\nCleaning complete.")
    for key, value in summary.items():
        print(f"  {key:<25}: {value}")
