from __future__ import annotations

import time
from typing import Any, Callable

import assemblyai as aai
from assemblyai import api as aai_api
from assemblyai.types import AssemblyAIError as SDKError, TranscriptResponse, TranscriptStatus

from ..transcription import RateLimitEvent, TranscriptData, TranscriptionError
from .retry import backoff_delay, rate_limit_delay, rate_limit_event


class AssemblyAIError(TranscriptionError):
    """Raised when AssemblyAI returns an unrecoverable error."""


class AssemblyAITranscriber:
    def __init__(
        self,
        api_key: str,
        *,
        poll_interval: float = 2.0,
        timeout: float | None = 3600.0,
        request_timeout: float = 30.0,
        max_retries: int = 5,
        on_rate_limit: Callable[[RateLimitEvent], None] | None = None,
    ) -> None:
        if not api_key:
            raise AssemblyAIError("Missing ASSEMBLYAI_API_KEY")

        aai.settings.api_key = api_key
        aai.settings.polling_interval = poll_interval
        aai.settings.http_timeout = request_timeout

        self.poll_interval = poll_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self._on_rate_limit = on_rate_limit

        self._client = aai.Client.get_default()
        self._transcriber = aai.Transcriber(
            client=self._client,
            config=aai.TranscriptionConfig(
                speaker_labels=True,
                format_text=True,
                punctuate=True,
                language_detection=True,
                speech_model=aai.SpeechModel.universal,
            ),
        )

    def transcribe(self, url: str) -> TranscriptData:
        submission = self._retry(lambda: self._transcriber.submit(url))
        transcript_id = submission.id
        if not transcript_id:
            raise AssemblyAIError("AssemblyAI response missing transcript id")

        start = time.monotonic()
        last_status: str | None = None

        while True:
            if self.timeout is not None and (time.monotonic() - start) > self.timeout:
                raise TimeoutError(
                    f"Timed out after {self.timeout:.0f}s (transcript {transcript_id}, last status: {last_status or 'unknown'})"
                )

            response = self._fetch_transcript(transcript_id)
            status = response.status.value if isinstance(response.status, TranscriptStatus) else str(response.status)
            last_status = status

            if status == TranscriptStatus.completed.value:
                data = response.dict()
                return TranscriptData(
                    transcription_id=transcript_id,
                    payload=data,
                )

            if status == TranscriptStatus.error.value:
                raise AssemblyAIError(response.error or "AssemblyAI error")

            time.sleep(self.poll_interval)

    def _fetch_transcript(self, transcript_id: str) -> TranscriptResponse:
        return self._retry(
            lambda: aai_api.get_transcript(self._client.http_client, transcript_id)
        )

    def _retry(self, func: Callable[[], Any]) -> Any:
        for attempt in range(1, self.max_retries + 1):
            try:
                return func()
            except SDKError as exc:
                response = self._client.last_response
                status = getattr(response, "status_code", None)
                if status in {429, 500, 502, 503, 504}:
                    delay = rate_limit_delay(response, attempt)
                    if status == 429 and self._on_rate_limit and response is not None:
                        endpoint = response.request.url.path if response.request else "unknown"
                        self._on_rate_limit(rate_limit_event(endpoint, delay, response))
                    time.sleep(delay)
                    continue
                raise AssemblyAIError(str(exc)) from exc
            except Exception as exc:  # pragma: no cover - unexpected errors
                if attempt == self.max_retries:
                    raise AssemblyAIError(str(exc)) from exc
                time.sleep(backoff_delay(attempt))

        raise AssemblyAIError("Maximum retries exceeded")
