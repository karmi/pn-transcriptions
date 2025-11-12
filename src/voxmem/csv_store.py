from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Sequence

from .util.path import normalize_to_dirname


REQUIRED_COLUMNS = ("filename", "url")
DEFAULT_COLUMNS = ("transcription_id", "status", "error")


@dataclass(slots=True)
class CsvRow:
    index: int
    data: dict[str, str]

    @property
    def filename(self) -> str:
        return self.data.get("filename", "").strip()

    @property
    def url(self) -> str:
        return self.data.get("url", "").strip()

    @property
    def transcription_id(self) -> str | None:
        value = self.data.get("transcription_id", "").strip()
        return value or None

    @property
    def status(self) -> str | None:
        value = self.data.get("status", "").strip()
        return value or None

    def is_completed(self) -> bool:
        return bool(self.transcription_id and (self.status == "completed"))


class CsvStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()
        self.fieldnames: list[str]
        self.rows: list[dict[str, str]]
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        with self.path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise ValueError("CSV file is missing headers")

            missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"CSV missing required columns: {', '.join(missing)}"
                )

            fieldnames = list(reader.fieldnames)
            for col in DEFAULT_COLUMNS:
                if col not in fieldnames:
                    fieldnames.append(col)

            self.fieldnames = fieldnames
            self.rows = [self._ensure_defaults(row) for row in reader]

    def _ensure_defaults(self, row: dict[str, str]) -> dict[str, str]:
        for col in DEFAULT_COLUMNS:
            row.setdefault(col, "")
        return row

    def slice(self, offset: int, limit: int | None) -> list[CsvRow]:
        if offset < 0:
            raise ValueError("Offset cannot be negative")

        start = min(offset, len(self.rows))
        end = len(self.rows) if limit is None else min(start + max(limit, 0), len(self.rows))
        return [CsvRow(index=i, data=self.rows[i]) for i in range(start, end)]

    def ensure_unique_filenames(self, rows: Sequence[CsvRow]) -> None:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for row in rows:
            name = row.filename
            try:
                norm = normalize_to_dirname(name)
            except ValueError as exc:
                raise ValueError(f"Row {row.index}: {exc}") from exc
            key = norm.lower()
            if key in seen:
                duplicates.add(name or norm)
            else:
                seen.add(key)

        if duplicates:
            raise ValueError(
                "Duplicate filenames detected: " + ", ".join(sorted(duplicates))
            )

    def pending(self, rows: Sequence[CsvRow]) -> list[CsvRow]:
        return [row for row in rows if not row.is_completed()]

    def mark_completed(self, index: int, transcription_id: str) -> None:
        with self._lock:
            entry = self.rows[index]
            entry["transcription_id"] = transcription_id
            entry["status"] = "completed"
            entry["error"] = ""
            self._flush_locked()

    def mark_failed(self, index: int, message: str) -> None:
        with self._lock:
            entry = self.rows[index]
            entry.setdefault("transcription_id", "")
            entry["status"] = "error"
            entry["error"] = message[:500]
            self._flush_locked()

    def _flush_locked(self) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.fieldnames)
                writer.writeheader()
                for row in self.rows:
                    writer.writerow({key: row.get(key, "") for key in self.fieldnames})
            os.replace(tmp_path, self.path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
