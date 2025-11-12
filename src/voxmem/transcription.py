from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import time
from typing import Any, Callable

import assemblyai as aai
from assemblyai import api as aai_api
from assemblyai.types import AssemblyAIError as SDKError, TranscriptResponse, TranscriptStatus


class AssemblyAIError(RuntimeError):
    """Raised when AssemblyAI returns an unrecoverable error."""


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
    vtt: str
    srt: str


class AssemblyAIClient:
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
                vtt = self._export_vtt(transcript_id)
                srt = self._export_srt(transcript_id)
                return TranscriptData(
                    transcription_id=transcript_id,
                    payload=data,
                    vtt=vtt,
                    srt=srt,
                )

            if status == TranscriptStatus.error.value:
                raise AssemblyAIError(response.error or "AssemblyAI error")

            time.sleep(self.poll_interval)

    def _fetch_transcript(self, transcript_id: str) -> TranscriptResponse:
        return self._retry(
            lambda: aai_api.get_transcript(self._client.http_client, transcript_id)
        )

    def _export_vtt(self, transcript_id: str) -> str:
        return self._retry(
            lambda: aai_api.export_subtitles_vtt(
                client=self._client.http_client,
                transcript_id=transcript_id,
                chars_per_caption=None,
            )
        )

    def _export_srt(self, transcript_id: str) -> str:
        return self._retry(
            lambda: aai_api.export_subtitles_srt(
                client=self._client.http_client,
                transcript_id=transcript_id,
                chars_per_caption=None,
            )
        )

    def _retry(self, func: Callable[[], Any]) -> Any:
        for attempt in range(1, self.max_retries + 1):
            try:
                return func()
            except SDKError as exc:
                response = self._client.last_response
                status = getattr(response, "status_code", None)
                if status in {429, 500, 502, 503, 504}:
                    delay = self._rate_limit_delay(response, attempt)
                    if status == 429 and self._on_rate_limit and response is not None:
                        endpoint = response.request.url.path if response.request else "unknown"
                        self._on_rate_limit(self._rate_limit_event(endpoint, delay, response))
                    time.sleep(delay)
                    continue
                raise AssemblyAIError(str(exc)) from exc
            except Exception as exc:  # pragma: no cover - unexpected errors
                if attempt == self.max_retries:
                    raise AssemblyAIError(str(exc)) from exc
                time.sleep(self._backoff_delay(attempt))

        raise AssemblyAIError("Maximum retries exceeded")

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        return min(2**attempt, 30)

    def _rate_limit_delay(self, response, attempt: int) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(float(retry_after), 1.0)
                except ValueError:
                    pass

            reset = response.headers.get("X-RateLimit-Reset")
            if reset:
                try:
                    reset_at = float(reset)
                    return max(reset_at - time.time(), 1.0)
                except ValueError:
                    pass

        return self._backoff_delay(attempt)

    def _rate_limit_event(self, endpoint: str, delay: float, response) -> RateLimitEvent:
        headers = response.headers if response is not None else {}
        limit = self._safe_int(headers.get("X-RateLimit-Limit"))
        remaining = self._safe_int(headers.get("X-RateLimit-Remaining"))

        reset_dt: dt.datetime | None = None
        reset = headers.get("X-RateLimit-Reset")
        if reset:
            try:
                reset_dt = dt.datetime.fromtimestamp(float(reset), tz=dt.timezone.utc)
            except ValueError:
                reset_dt = None

        return RateLimitEvent(
            endpoint=endpoint,
            delay=delay,
            limit=limit,
            remaining=remaining,
            reset_at=reset_dt,
        )

    @staticmethod
    def _safe_int(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
