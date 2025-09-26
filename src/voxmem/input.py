from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, Tuple
from urllib.parse import urlparse

AUDIO_EXTS: Tuple[str, ...] = (".mp3", ".wav", ".m4a")


class Input(Protocol):
    def __iter__(self) -> Iterator[str]: ...


@dataclass
class LocalInput:
    path: Path
    exts: Tuple[str, ...] = AUDIO_EXTS

    def __iter__(self) -> Iterator[str]:
        p = Path(self.path)
        if p.is_dir():
            for ext in self.exts:
                yield from (str(f) for f in p.rglob(f"*{ext}") if f.is_file())
        elif p.is_file() and p.suffix.lower() in self.exts:
            yield str(p)


@dataclass
class R2Input:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    prefix: str = ""
    exts: Tuple[str, ...] = AUDIO_EXTS
    presign_ttl: int = 4 * 60 * 60
    endpoint: str | None = None
    list_timeout: float | None = 3600  # Safety "hatch"

    @staticmethod
    def from_env(token: str, presign_ttl: int = 4 * 60 * 60) -> "R2Input":
        parsed = urlparse(token)
        if parsed.scheme != "r2":
            raise ValueError(f"Invalid R2 token (missing r2://): {token}")

        bucket = parsed.netloc
        if not bucket:
            raise ValueError(f"Invalid R2 token (no bucket): {token}")

        prefix = parsed.path.lstrip("/") if parsed.path else ""

        account_id = os.environ.get("R2_ACCOUNT_ID")
        access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
        secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        endpoint = os.environ.get("R2_ENDPOINT")

        missing = [
            k
            for k, v in [
                ("R2_ACCOUNT_ID", account_id),
                ("R2_ACCESS_KEY_ID", access_key_id),
                ("R2_SECRET_ACCESS_KEY", secret_access_key),
            ]
            if not v
        ]
        if missing:
            raise RuntimeError(
                "Missing Cloudflare R2 credentials: " + ", ".join(missing)
            )

        return R2Input(
            account_id=account_id,  # type: ignore[arg-type]
            access_key_id=access_key_id,  # type: ignore[arg-type]
            secret_access_key=secret_access_key,  # type: ignore[arg-type]
            bucket=bucket,
            prefix=prefix,
            presign_ttl=presign_ttl,
            endpoint=endpoint,
        )

    def _client(self):
        import boto3
        from botocore.config import Config

        endpoint = (
            self.endpoint or f"https://{self.account_id}.r2.cloudflarestorage.com"
        )
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )

    def __iter__(self) -> Iterator[str]:
        s3 = self._client()
        kwargs: dict = {"Bucket": self.bucket}
        if self.prefix:
            kwargs["Prefix"] = self.prefix

        start = time.monotonic()

        while True:
            if (
                self.list_timeout is not None
                and (time.monotonic() - start) > self.list_timeout
            ):
                raise TimeoutError(
                    f"Timed out listing r2://{self.bucket}/{self.prefix}"
                )

            resp = s3.list_objects_v2(**kwargs)

            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith(self.exts):
                    url = s3.generate_presigned_url(
                        ClientMethod="get_object",
                        Params={"Bucket": self.bucket, "Key": key},
                        ExpiresIn=self.presign_ttl,
                    )
                    yield url

            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
                if not token:
                    # Safety: avoid infinite loop if API says truncated but no token provided
                    break
                kwargs["ContinuationToken"] = token
            else:
                break
