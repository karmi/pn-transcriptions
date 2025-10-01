from __future__ import annotations

import numbers
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
    timeout: float | None = 3600  # seconds; None = no limit

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

        start = time.monotonic()
        last_status: str | None = None

        while True:
            if self.timeout is not None and (time.monotonic() - start) > self.timeout:
                raise TimeoutError(
                    f"Timed out after {self.timeout:.0f}s "
                    f"(transcript {submission.id}, last status: {last_status or 'unknown'})"
                )

            tr = aai.Transcript.get_by_id(submission.id)
            status = getattr(tr.status, "value", str(tr.status))
            last_status = status

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


@dataclass
class MLXWhisperTranscriber:
    model_id: str = "mlx-community/whisper-large-v3-mlx"
    language: str | None = None
    verbose: bool = False

    def transcribe(self, source: str) -> Mapping[str, Any]:
        try:
            import mlx_whisper
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "mlx-whisper is required for the MLX backend; install it via 'pip install mlx-whisper'."
            ) from exc

        kwargs: dict[str, Any] = {
            "path_or_hf_repo": self.model_id,
            "word_timestamps": True,
        }

        if self.language:
            kwargs["language"] = self.language
        if self.verbose is not None:
            kwargs["verbose"] = self.verbose

        result = mlx_whisper.transcribe(source, **kwargs)

        segments = result.get("segments") or []
        words: list[dict[str, Any]] = []
        audio_duration: float | None = None

        for segment in segments:
            seg_words = segment.get("words") or []
            for word in seg_words:
                cleaned = {
                    key: self._to_builtin(value)
                    for key, value in word.items()
                }
                words.append(cleaned)

                end = cleaned.get("end")
                if isinstance(end, (int, float)):
                    audio_duration = (
                        max(audio_duration or 0.0, float(end))
                        if audio_duration is not None
                        else float(end)
                    )

            if not seg_words:
                end = self._to_builtin(segment.get("end"))
                if isinstance(end, (int, float)):
                    audio_duration = (
                        max(audio_duration or 0.0, float(end))
                        if audio_duration is not None
                        else float(end)
                    )

        return {
            "id": None,
            "status": "completed",
            "error": None,
            "confidence": None,
            "audio_duration": audio_duration,
            "text": result.get("text"),
            "words": words,
        }

    @staticmethod
    def _to_builtin(value: Any) -> Any:
        if isinstance(value, numbers.Number):
            return float(value)
        if isinstance(value, (list, tuple)):
            return [MLXWhisperTranscriber._to_builtin(v) for v in value]
        if isinstance(value, dict):
            return {
                k: MLXWhisperTranscriber._to_builtin(v) for k, v in value.items()
            }
        return value
