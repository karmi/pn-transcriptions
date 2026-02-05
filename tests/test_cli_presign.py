from __future__ import annotations

import logging
import threading

import pytest

from voxmem.util import url_signing
from voxmem.transcription import TranscriptionError


class DummyGetResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def close(self) -> None:
        return None


class DummyS3Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str], int]] = []

    def generate_presigned_url(self, *, ClientMethod: str, Params: dict[str, str], ExpiresIn: int) -> str:
        self.calls.append((ClientMethod, Params, ExpiresIn))
        return "https://signed.example/audio.mp3?X-Amz-Signature=secret"


def test_resolve_audio_url_presigns_and_preflights(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_client = DummyS3Client()

    def fake_boto3_client(*args, **kwargs):
        return dummy_client

    def fake_get(
        url: str,
        *,
        headers: dict[str, str],
        allow_redirects: bool,
        timeout: float,
        stream: bool,
    ):
        return DummyGetResponse(206)

    monkeypatch.setattr(url_signing.boto3, "client", fake_boto3_client)
    monkeypatch.setattr(url_signing.requests, "get", fake_get)
    monkeypatch.setenv("CLOUDFLARE_R2_ENDPOINT", "https://example.r2")
    monkeypatch.setenv("CLOUDFLARE_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("CLOUDFLARE_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("R2_SIGN_TTL_SECONDS", raising=False)
    monkeypatch.setenv("R2_SIGN_PREFLIGHT", "1")
    monkeypatch.setenv("R2_SIGN_PREFLIGHT_METHOD", "GET")

    logger = logging.getLogger("test")
    url = url_signing.resolve_audio_url("s3://bucket/key.mp3", logger, threading.local())

    assert url.startswith("https://signed.example/")
    assert dummy_client.calls == [
        ("get_object", {"Bucket": "bucket", "Key": "key.mp3"}, 604800)
    ]


def test_resolve_audio_url_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_boto3_client(*args, **kwargs):
        raise AssertionError("boto3 client should not be called")

    def fake_get(
        url: str,
        *,
        headers: dict[str, str],
        allow_redirects: bool,
        timeout: float,
        stream: bool,
    ):
        raise AssertionError("requests.get should not be called")

    monkeypatch.setattr(url_signing.boto3, "client", fake_boto3_client)
    monkeypatch.setattr(url_signing.requests, "get", fake_get)

    logger = logging.getLogger("test")
    url = url_signing.resolve_audio_url(
        "https://example.com/audio.mp3", logger, threading.local()
    )

    assert url == "https://example.com/audio.mp3"


def test_resolve_audio_url_preflight_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_client = DummyS3Client()

    def fake_boto3_client(*args, **kwargs):
        return dummy_client

    def fake_get(
        url: str,
        *,
        headers: dict[str, str],
        allow_redirects: bool,
        timeout: float,
        stream: bool,
    ):
        return DummyGetResponse(403)

    monkeypatch.setattr(url_signing.boto3, "client", fake_boto3_client)
    monkeypatch.setattr(url_signing.requests, "get", fake_get)
    monkeypatch.setenv("CLOUDFLARE_R2_ENDPOINT", "https://example.r2")
    monkeypatch.setenv("CLOUDFLARE_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("CLOUDFLARE_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_SIGN_PREFLIGHT", "1")
    monkeypatch.setenv("R2_SIGN_PREFLIGHT_METHOD", "GET")

    logger = logging.getLogger("test")
    with pytest.raises(TranscriptionError):
        url_signing.resolve_audio_url("s3://bucket/key.mp3", logger, threading.local())
