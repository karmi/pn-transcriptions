from __future__ import annotations

import datetime as dt
import logging
import os
import threading
from urllib.parse import parse_qsl, urlparse, urlunparse

import boto3
from botocore.config import Config
import requests

from ..transcription import TranscriptionError


def resolve_audio_url(
    url: str,
    logger: logging.Logger,
    thread_local: threading.local,
    *,
    preflight: bool | None = None,
) -> str:
    if not url:
        raise TranscriptionError("Missing URL")
    if not is_storage_url(url):
        return url

    bucket, key = parse_storage_url(url)
    s3 = _get_storage_client(thread_local)
    ttl_seconds = _get_sign_ttl_seconds(logger)
    presigned = _presign_url(s3, bucket=bucket, key=key, ttl_seconds=ttl_seconds, method="GET")
    expires_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=ttl_seconds)

    if preflight is None:
        preflight = os.environ.get("R2_SIGN_PREFLIGHT", "1").strip() != "0"
    if preflight:
        timeout = _get_preflight_timeout()
        method = _get_preflight_method()
        if method == "HEAD":
            preflight_url = _presign_url(
                s3, bucket=bucket, key=key, ttl_seconds=ttl_seconds, method="HEAD"
            )
        else:
            preflight_url = presigned
        status = _preflight_status(preflight_url, timeout=timeout, method=method)
        if status not in {200, 206}:
            raise TranscriptionError(
                f"Signed URL preflight failed for {bucket}/{key} (status {status})"
            )

    redacted = _redact_presigned_url(presigned)
    logger.info(
        "signed_url bucket=%s key=%s expires_at=%s url=%s",
        bucket,
        key,
        expires_at.isoformat(),
        redacted,
    )
    return presigned


def resolve_head_url(
    url: str,
    logger: logging.Logger,
    thread_local: threading.local,
) -> str:
    if not url:
        raise TranscriptionError("Missing URL")
    if not is_storage_url(url):
        return url

    bucket, key = parse_storage_url(url)
    s3 = _get_storage_client(thread_local)
    ttl_seconds = _get_sign_ttl_seconds(logger)
    presigned = _presign_url(s3, bucket=bucket, key=key, ttl_seconds=ttl_seconds, method="HEAD")
    expires_at = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=ttl_seconds)

    redacted = _redact_presigned_url(presigned)
    logger.info(
        "signed_head_url bucket=%s key=%s expires_at=%s url=%s",
        bucket,
        key,
        expires_at.isoformat(),
        redacted,
    )
    return presigned


def is_storage_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"s3", "r2"}


def parse_storage_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path or parsed.path == "/":
        raise TranscriptionError(f"Invalid s3/r2 URL: {url}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def _get_sign_ttl_seconds(logger: logging.Logger) -> int:
    raw = os.environ.get("R2_SIGN_TTL_SECONDS", "").strip()
    if not raw:
        return 604800
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid R2_SIGN_TTL_SECONDS=%s; using 604800", raw)
        return 604800
    if value > 604800:
        logger.warning("R2_SIGN_TTL_SECONDS capped to 604800 (requested %s)", value)
        return 604800
    if value <= 0:
        logger.warning("R2_SIGN_TTL_SECONDS must be >0; using 604800")
        return 604800
    return value


def _get_preflight_timeout() -> float:
    raw = os.environ.get("R2_SIGN_PREFLIGHT_TIMEOUT", "").strip()
    if not raw:
        return 30.0
    try:
        value = float(raw)
    except ValueError:
        return 30.0
    return max(1.0, value)


def _get_preflight_method() -> str:
    return os.environ.get("R2_SIGN_PREFLIGHT_METHOD", "HEAD").strip().upper()


def _get_storage_client(thread_local: threading.local):
    client = getattr(thread_local, "storage_client", None)
    if client is not None:
        return client
    endpoint = os.environ.get("CLOUDFLARE_R2_ENDPOINT", "").strip()
    access_key = os.environ.get("CLOUDFLARE_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("CLOUDFLARE_SECRET_ACCESS_KEY", "").strip()
    if not endpoint or not access_key or not secret_key:
        raise TranscriptionError(
            "Missing CLOUDFLARE_R2_ENDPOINT/CLOUDFLARE_ACCESS_KEY_ID/CLOUDFLARE_SECRET_ACCESS_KEY"
        )
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    thread_local.storage_client = client
    return client


def _preflight_status(url: str, *, timeout: float, method: str) -> int:
    try:
        if method == "GET":
            response = requests.get(
                url,
                headers={"Range": "bytes=0-0"},
                allow_redirects=True,
                timeout=timeout,
                stream=True,
            )
        else:
            response = requests.head(url, allow_redirects=True, timeout=timeout)
    except requests.RequestException:
        return 0
    try:
        return response.status_code
    finally:
        response.close()


def _redact_presigned_url(url: str) -> str:
    allow_full = os.environ.get("R2_SIGN_LOG_FULL_URLS", "").strip() == "1"
    if allow_full:
        return url
    parsed = urlparse(url)
    query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() == "x-amz-signature":
            query.append((key, "REDACTED"))
        else:
            query.append((key, value))
    redacted = parsed._replace(query="&".join(f"{k}={v}" for k, v in query))
    return urlunparse(redacted)


def _presign_url(s3, *, bucket: str, key: str, ttl_seconds: int, method: str) -> str:
    if method == "HEAD":
        client_method = "head_object"
    else:
        client_method = "get_object"
    return s3.generate_presigned_url(
        ClientMethod=client_method,
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=ttl_seconds,
    )
