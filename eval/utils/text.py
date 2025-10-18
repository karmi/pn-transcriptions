"""Text normalization utilities shared across evaluation notebooks."""

from __future__ import annotations

import re

NORMALIZE_PATTERN = re.compile(
    r"""
    [()“”„"]            # parentheses and quotation marks
  | –                   # standalone en dash
  | (?<!\d),(?!\d)      # commas not between digits
  | :(?!\d)             # colons not followed by a digit (keep 3:2, etc.)
  | [!?;]               # other sentence-ending punctuation to drop
    """,
    re.VERBOSE,
)


def normalize(text: str) -> str:
    """Normalize text to align with reference transcripts."""
    cleaned = text.replace("\u00a0", " ").lower()
    cleaned = NORMALIZE_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()  # remove double spaces
    while cleaned.endswith("."):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


__all__ = ["normalize", "NORMALIZE_PATTERN"]
