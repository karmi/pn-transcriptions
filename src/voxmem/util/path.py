from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from unidecode import unidecode


_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def normalize_to_dirname(name: str, max_length: int = 100) -> str:
    if not name:
        raise ValueError("Filename must be provided for normalization")

    filename = Path(name).name.strip()
    if not filename:
        raise ValueError("Filename must contain at least one visible character")

    base = Path(filename).stem.strip() or filename

    def _clean(value: str) -> str:
        text = unicodedata.normalize("NFKD", value)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.replace(" ", "_")
        text = _SAFE_PATTERN.sub("_", text)
        text = re.sub(r"_+", "_", text)
        return text.strip("._-")

    candidate = _clean(unidecode(base))
    if not candidate:
        raise ValueError("Filename cannot be normalized; specify a safer name")

    if max_length > 0 and len(candidate) > max_length:
        candidate = candidate[:max_length]

    return candidate
