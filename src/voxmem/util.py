from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def name_from_source(source: str) -> str:
    if source.startswith(("http://", "https://")):
        path = urlparse(source).path.rsplit("/", 1)[-1]
        return Path(path).stem or "output"
    return Path(source).stem or "output"
