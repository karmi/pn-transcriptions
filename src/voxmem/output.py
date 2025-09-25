from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Mapping, Any
import json


class Output(Protocol):
    def save(self, name: str, data: Mapping[str, Any]) -> str: ...


@dataclass
class LocalJsonOutput:
    root: Path

    def save(self, name: str, data: Mapping[str, Any]) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{name}.json"
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)
