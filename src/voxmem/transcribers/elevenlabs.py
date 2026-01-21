from __future__ import annotations

import json
import time
from typing import Any, Callable

import requests

from ..transcription import (
    RateLimitEvent,
    TranscriptData,
    TranscriptionError,
)
from .retry import backoff_delay, rate_limit_delay, rate_limit_event


class ElevenLabsError(TranscriptionError):
    """Raised when ElevenLabs returns an unrecoverable error."""


def _payload_has_words(payload: dict[str, Any]) -> bool:
    if "transcripts" in payload:
        for transcript in payload.get("transcripts") or []:
            if transcript.get("words"):
                return True
        return False
    return bool(payload.get("words"))


def _format_form_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


class ElevenLabsTranscriber:
    API_URL = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(
        self,
        api_key: str,
        *,
        model_id: str = "scribe_v2",
        request_timeout: float = 3600.0,
        timeout: float | None = 3600.0,
        max_retries: int = 5,
        diarize: bool = True,
        tag_audio_events: bool = True,
        timestamps_granularity: str = "word",
        num_speakers: int | None = None,
        poll_until_ready: bool = True,
        poll_interval: float = 2.0,
        session: requests.Session | None = None,
        on_rate_limit: Callable[[RateLimitEvent], None] | None = None,
    ) -> None:
        if not api_key:
            raise ElevenLabsError("Missing ELEVENLABS_API_KEY")

        self.api_key = api_key
        self.model_id = model_id
        self.request_timeout = request_timeout
        self.timeout = timeout
        self.max_retries = max_retries
        self.diarize = diarize
        self.tag_audio_events = tag_audio_events
        self.timestamps_granularity = timestamps_granularity
        self.num_speakers = num_speakers
        self.poll_until_ready = poll_until_ready
        self.poll_interval = poll_interval
        self._session = session or requests.Session()
        self._on_rate_limit = on_rate_limit

    def transcribe(self, url: str) -> TranscriptData:
        payload = self._request_transcription(url)
        transcription_id = str(payload.get("transcription_id") or "").strip()
        if not transcription_id:
            raise ElevenLabsError("ElevenLabs response missing transcription id")

        if not _payload_has_words(payload):
            if not self.poll_until_ready:
                raise ElevenLabsError(
                    "ElevenLabs response missing transcript payload; enable polling to wait for completion."
                )
            payload = self._poll_transcript(transcription_id)

        return TranscriptData(
            transcription_id=transcription_id,
            payload=dict(payload),
        )

    def _request_transcription(self, url: str) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "model_id": self.model_id,
            "cloud_storage_url": url,
            "timestamps_granularity": self.timestamps_granularity,
            "diarize": self.diarize,
            "tag_audio_events": self.tag_audio_events,
        }
        if self.num_speakers is not None:
            fields["num_speakers"] = self.num_speakers

        files = {
            name: (None, _format_form_value(value)) for name, value in fields.items()
        }
        headers = {"xi-api-key": self.api_key}

        for attempt in range(1, self.max_retries + 1):
            response = None
            try:
                response = self._session.post(
                    self.API_URL,
                    headers=headers,
                    files=files,
                    timeout=self.request_timeout,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise ElevenLabsError(str(exc)) from exc
                time.sleep(backoff_delay(attempt))
                continue

            if response.status_code in {429, 500, 502, 503, 504}:
                delay = rate_limit_delay(response, attempt)
                if response.status_code == 429 and self._on_rate_limit:
                    endpoint = response.request.url if response.request else self.API_URL
                    self._on_rate_limit(rate_limit_event(endpoint, delay, response))
                time.sleep(delay)
                continue

            if response.status_code >= 400:
                detail = response.text.strip()
                message = f"ElevenLabs error {response.status_code}"
                if detail:
                    message = f"{message}: {detail}"
                raise ElevenLabsError(message)

            try:
                return response.json()
            except ValueError as exc:
                raise ElevenLabsError("ElevenLabs response was not valid JSON") from exc

        raise ElevenLabsError("Maximum retries exceeded")

    def _poll_transcript(self, transcription_id: str) -> dict[str, Any]:
        start = time.monotonic()
        last_error: str | None = None

        while True:
            if self.timeout is not None and (time.monotonic() - start) > self.timeout:
                detail = f"last error: {last_error}" if last_error else "no response yet"
                raise TimeoutError(
                    f"Timed out after {self.timeout:.0f}s (transcript {transcription_id}, {detail})"
                )

            try:
                payload = self._fetch_transcript(transcription_id)
            except ElevenLabsError as exc:
                last_error = str(exc)
                time.sleep(self.poll_interval)
                continue

            if payload is None:
                time.sleep(self.poll_interval)
                continue

            if not _payload_has_words(payload):
                time.sleep(self.poll_interval)
                continue

            return payload

    def _fetch_transcript(self, transcription_id: str) -> dict[str, Any] | None:
        url = f"{self.API_URL}/transcripts/{transcription_id}"
        headers = {"xi-api-key": self.api_key}

        for attempt in range(1, self.max_retries + 1):
            response = None
            try:
                response = self._session.get(
                    url,
                    headers=headers,
                    timeout=self.request_timeout,
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise ElevenLabsError(str(exc)) from exc
                time.sleep(backoff_delay(attempt))
                continue

            if response.status_code == 404:
                return None

            if response.status_code in {429, 500, 502, 503, 504}:
                delay = rate_limit_delay(response, attempt)
                if response.status_code == 429 and self._on_rate_limit:
                    endpoint = response.request.url if response.request else url
                    self._on_rate_limit(rate_limit_event(endpoint, delay, response))
                time.sleep(delay)
                continue

            if response.status_code >= 400:
                detail = response.text.strip()
                message = f"ElevenLabs error {response.status_code}"
                if detail:
                    message = f"{message}: {detail}"
                raise ElevenLabsError(message)

            try:
                return response.json()
            except ValueError as exc:
                raise ElevenLabsError("ElevenLabs response was not valid JSON") from exc

        raise ElevenLabsError("Maximum retries exceeded")
