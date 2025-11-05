from __future__ import annotations

import re

NORMALIZE_PATTERN = re.compile(
    r"""
    [()“”„"]            # parentheses and quotation marks
  | –                   # standalone en dash
  | (?<!\d),(?!\d)      # commas not between digits
  | (?<!\d)\.(?!\d)     # periods not between digits
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


if __name__ == "__main__":  # simple inline regression checks
    assert normalize("Satelit. Start.") == "satelit start"
    assert normalize("Stálo 3.5 Kč.") == "stálo 3.5 kč"
    assert normalize("Finále bylo 3:2.") == "finále bylo 3:2"
