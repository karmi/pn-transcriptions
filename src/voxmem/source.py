from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Source(Protocol):
    def resolve(self, path: str) -> str: ...


@dataclass
class LocalSource:
    root: Path | None = None

    def resolve(self, path: str) -> str:
        p = Path(self.root, path) if self.root else Path(path)
        return str(p)
