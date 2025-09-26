from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import assemblyai as aai


class Transcriber(Protocol):
    def transcribe(self, source: str) -> Mapping[str, Any]: ...


@dataclass
class NoopTranscriber:
    def transcribe(self, source: str) -> Mapping[str, Any]:
        return {"noop": f"would transcribe {source}"}


@dataclass
class AssemblyAITranscriber:
    api_key: str | None = None
    polling_interval: float = 2.0

    def __post_init__(self) -> None:
        aai.settings.api_key = self.api_key or aai.settings.api_key
        if not aai.settings.api_key:
            raise RuntimeError("Missing ASSEMBLYAI_API_KEY")
        aai.settings.polling_interval = self.polling_interval
        self._config = aai.TranscriptionConfig(
            speaker_labels=True,
            format_text=True,
            punctuate=True,
            speech_model=aai.SpeechModel.best,
            language_detection=True,
        )

    def transcribe(self, source: str) -> Mapping[str, Any]:
        transcriber = aai.Transcriber(config=self._config)
        submission = transcriber.submit(source)

        while True:
            tr = aai.Transcript.get_by_id(submission.id)
            status = getattr(tr.status, "value", str(tr.status))
            if status == "completed":
                raw = getattr(tr, "json_response", None)
                data = raw() if callable(raw) else raw
                if not isinstance(data, dict):
                    return data

                keys = (
                    "id",
                    "status",
                    "error",
                    "confidence",
                    "audio_duration",
                    "text",
                    "words",
                )
                return {k: data.get(k) for k in keys}
            if status == "error":
                err = getattr(tr, "error", None) or "AssemblyAI transcription error"
                raise RuntimeError(err)
            time.sleep(aai.settings.polling_interval)
