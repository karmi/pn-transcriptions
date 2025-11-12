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
    vtt_path: Path | None
    srt_path: Path | None


class TranscriptStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bundle(
        self,
        filename: str,
        transcription_id: str,
        payload: Mapping[str, Any],
        *,
        vtt: str | None = None,
        srt: str | None = None,
    ) -> StorageResult:
        folder = self._ensure_folder(filename)
        json_path = self._write_json(folder / f"{transcription_id}.json", payload)
        vtt_path = self._write_text(folder / f"{transcription_id}.vtt", vtt)
        srt_path = self._write_text(folder / f"{transcription_id}.srt", srt)
        return StorageResult(folder=folder, json_path=json_path, vtt_path=vtt_path, srt_path=srt_path)

    def _ensure_folder(self, filename: str) -> Path:
        safe = normalize_to_dirname(filename)
        folder = self.root / safe
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _write_json(self, target: Path, payload: Mapping[str, Any]) -> Path:
        self._atomic_write(target, json.dumps(payload, ensure_ascii=False, indent=2))
        return target

    def _write_text(self, target: Path, content: str | None) -> Path | None:
        if content is None:
            return None
        self._atomic_write(target, content)
        return target

    def _atomic_write(self, target: Path, data: str) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(data)
            os.replace(tmp_path, target)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
