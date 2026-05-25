"""
Character-level tokenizer.

Maps every unique character in a text corpus to an integer index.
No external tokenization library is used.
"""

import json


class CharTokenizer:
    """A simple character-level tokenizer.

    Each unique character in the training text receives a unique integer ID.
    Unknown characters encountered at inference time are silently skipped.
    """

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char_to_idx: dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char: dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of integer token IDs."""
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices: list[int]) -> str:
        """Convert a list of integer token IDs back to a string."""
        return "".join(self.idx_to_char[i] for i in indices if i in self.idx_to_char)

    def save(self, path: str) -> None:
        """Persist the vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"chars": self.chars}, fh)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Restore a tokenizer from a saved JSON vocabulary file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        obj = cls.__new__(cls)
        obj.chars = data["chars"]
        obj.vocab_size = len(obj.chars)
        obj.char_to_idx = {ch: i for i, ch in enumerate(obj.chars)}
        obj.idx_to_char = {i: ch for i, ch in enumerate(obj.chars)}
        return obj
