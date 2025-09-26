from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


class Output(Protocol):
    def save(self, name: str, data: Mapping[str, Any]) -> str: ...


@dataclass
class LocalJsonOutput:
    root: Path

    def path_for(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def exists(self, name: str) -> bool:
        return self.path_for(name).exists()

    def save(self, name: str, data: Mapping[str, Any]) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(name)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)
