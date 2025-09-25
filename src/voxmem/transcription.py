from __future__ import annotations

import time

from dataclasses import dataclass
from typing import Protocol, Optional


class Transcriber(Protocol):
    def transcribe(self, source: str, *, out_dir: str) -> str: ...


@dataclass
class NoopTranscriber:
    def transcribe(self, source: str, *, out_dir: str) -> str:
        time.sleep(1)
        return f"NOOP: would transcribe {source} → {out_dir}"
