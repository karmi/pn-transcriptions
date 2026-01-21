from __future__ import annotations

from typing import Any

from voxmem.transcription import TranscriptData
from voxmem.transcribers.elevenlabs import ElevenLabsTranscriber


class DummyRequest:
    def __init__(self, url: str) -> None:
        self.url = url


class DummyResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.text = "ok"
        self.request = None

    def json(self) -> dict[str, Any]:
        return self._payload


class DummySession:
    def __init__(
        self,
        post_responses: list[DummyResponse],
        get_responses: list[DummyResponse] | None = None,
    ) -> None:
        self.post_responses = post_responses
        self.get_responses = get_responses or []
        self.last_post_url: str | None = None
        self.last_post_headers: dict[str, str] | None = None
        self.last_post_files: dict[str, tuple[None, str]] | None = None
        self.last_post_timeout: float | None = None
        self.last_get_url: str | None = None
        self.last_get_headers: dict[str, str] | None = None
        self.last_get_timeout: float | None = None
        self.last_get_params: dict[str, str] | None = None
        self.get_calls = 0

    def post(self, url: str, *, headers: dict[str, str], files, timeout: float):
        self.last_post_url = url
        self.last_post_headers = headers
        self.last_post_files = files
        self.last_post_timeout = timeout
        response = self.post_responses.pop(0)
        response.request = DummyRequest(url)
        return response

    def get(self, url: str, *, headers: dict[str, str], timeout: float, params=None):
        self.get_calls += 1
        self.last_get_url = url
        self.last_get_headers = headers
        self.last_get_timeout = timeout
        self.last_get_params = params
        response = self.get_responses.pop(0)
        response.request = DummyRequest(url)
        return response


def test_elevenlabs_transcriber_builds_request_and_response() -> None:
    payload = {
        "transcription_id": "abc123",
        "text": "Hello world",
        "words": [
            {"text": "Hello", "start": 0.0, "end": 0.5, "type": "word"},
            {"text": " ", "type": "spacing"},
            {"text": "world", "start": 0.6, "end": 1.0, "type": "word"},
        ],
    }
    response = DummyResponse(payload)
    session = DummySession([response])

    client = ElevenLabsTranscriber(
        api_key="test-key",
        session=session,
        request_timeout=12.0,
    )
    result = client.transcribe("https://example.com/audio.mp3")

    assert isinstance(result, TranscriptData)
    assert result.transcription_id == "abc123"
    assert result.payload["text"] == "Hello world"

    assert session.last_post_url == ElevenLabsTranscriber.API_URL
    assert session.last_post_headers == {"xi-api-key": "test-key"}
    assert session.last_post_timeout == 12.0
    assert session.last_post_files is not None
    assert session.last_post_files["cloud_storage_url"] == (
        None,
        "https://example.com/audio.mp3",
    )
    assert session.last_post_files["model_id"] == (None, "scribe_v2")
    assert session.last_get_url is None


def test_elevenlabs_transcriber_polls_when_batch_response_incomplete() -> None:
    initial = DummyResponse({"transcription_id": "xyz", "message": "queued"})
    followup = DummyResponse(
        {
            "transcription_id": "xyz",
            "text": "done",
            "words": [
                {"text": "done", "start": 0.0, "end": 0.2, "type": "word"},
            ],
        }
    )
    session = DummySession([initial], [followup])

    client = ElevenLabsTranscriber(
        api_key="test-key",
        session=session,
        request_timeout=5.0,
        poll_interval=0.0,
        timeout=5.0,
    )
    result = client.transcribe("https://example.com/audio.mp3")

    assert result.transcription_id == "xyz"
    assert session.get_calls == 1
