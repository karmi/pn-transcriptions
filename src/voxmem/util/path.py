from __future__ import annotations

import re
import unicodedata
from pathlib import Path


_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def normalize_to_dirname(name: str, max_length: int = 100) -> str:
    if not name:
        raise ValueError("Filename must be provided for normalization")

    filename = Path(name).name.strip()
    if not filename:
        raise ValueError("Filename must contain at least one visible character")

    stem = Path(filename).stem.strip()
    candidate = stem or filename

    candidate = unicodedata.normalize("NFKD", candidate)
    candidate = candidate.encode("ascii", "ignore").decode("ascii")
    candidate = candidate.replace(" ", "_")
    candidate = _SAFE_PATTERN.sub("_", candidate)
    candidate = candidate.strip("._-")

    if not candidate:
        raise ValueError("Filename cannot be normalized; specify a safer name")

    if max_length > 0 and len(candidate) > max_length:
        candidate = candidate[:max_length]

    return candidate
