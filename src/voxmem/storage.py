from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .util.path import normalize_to_dirname


@dataclass(slots=True)
class StorageResult:
    folder: Path
    json_path: Path


class TranscriptStorage:
    def __init__(self, root: Path, file_mode: int = 0o644) -> None:
        self.root = root
        self.file_mode = file_mode
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bundle(
        self,
        filename: str,
        transcription_id: str,
        payload: Mapping[str, Any],
    ) -> StorageResult:
        folder = self._ensure_folder(filename)
        json_path = self._write_json(folder / f"{transcription_id}.json", payload)
        return StorageResult(folder=folder, json_path=json_path)

    def _ensure_folder(self, filename: str) -> Path:
        safe = normalize_to_dirname(filename)
        folder = self.root / safe
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _write_json(self, target: Path, payload: Mapping[str, Any]) -> Path:
        self._atomic_write(target, json.dumps(payload, ensure_ascii=False, indent=2))
        return target

    def _atomic_write(self, target: Path, data: str) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(data)
            os.chmod(tmp_path, self.file_mode)
            os.replace(tmp_path, target)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
