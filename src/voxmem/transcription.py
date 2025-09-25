from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional


class Transcriber(Protocol):
    def transcribe(self, source: str, *, out_dir: str) -> str: ...


@dataclass
class NoopTranscriber:
    def transcribe(self, source: str, *, out_dir: str) -> str:
        return f"NOOP: would transcribe {source} → {out_dir}"
