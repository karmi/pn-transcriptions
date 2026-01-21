from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from typing import Any, Protocol


class TranscriptionError(RuntimeError):
    """Raised when a transcription provider returns an unrecoverable error."""


@dataclass(slots=True)
class RateLimitEvent:
    endpoint: str
    delay: float
    limit: int | None
    remaining: int | None
    reset_at: dt.datetime | None


@dataclass(slots=True)
class TranscriptData:
    transcription_id: str
    payload: dict[str, Any]


class Transcriber(Protocol):
    def transcribe(self, url: str) -> TranscriptData:
        """Transcribe the audio at the given URL."""
