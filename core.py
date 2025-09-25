from __future__ import annotations

import os
import time
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor, Future

import assemblyai as aai
import boto3
from botocore.config import Config
from tqdm import tqdm


def init_assemblyai(
    api_key: Optional[str] = None,
    polling_interval: float = 2.0,
) -> aai.Transcriber:
    aai.settings.api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not aai.settings.api_key:
        raise RuntimeError("Missing ASSEMBLYAI_API_KEY")
    aai.settings.polling_interval = (
        polling_interval  # SDK default is ~3s; we use 2s for snappier bars
    )

    config = aai.TranscriptionConfig(
        speaker_labels=True,
        format_text=True,
        punctuate=True,
        speech_model=aai.SpeechModel.best,
        language_detection=True,
    )

    return aai.Transcriber(config=config)


def poll_with_progress(
    transcriber: aai.Transcriber, transcript_id: str, desc: str
) -> aai.Transcript:
    frames = "|/-\\"  # ASCII spinner for widest compatibility
    idx = 0
    status = "queued"
    poll_every = aai.settings.polling_interval
    start = time.monotonic()
    last_poll = start  # delay first API call so spinner renders immediately
    max_len = 0

    def fmt_elapsed(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        return f"{m:02d}:{s:02d}"

    try:
        while True:
            now = time.monotonic()

            # Poll the API at the configured interval
            if now - last_poll >= poll_every:
                try:
                    transcript = aai.Transcript.get_by_id(transcript_id)
                    status = getattr(transcript.status, "value", str(transcript.status))
                    if status in ("completed", "error"):
                        line = f"{desc}  {status}  [{fmt_elapsed(now - start)}]"
                        pad = " " * max(0, max_len - len(line))
                        sys.stdout.write("\r" + line + pad + "\n")
                        sys.stdout.flush()
                        return transcript
                except Exception:
                    status = "retrying…"
                finally:
                    last_poll = now

            # Animate spinner between polls
            frame = frames[idx % len(frames)]
            idx += 1
            line = f"{desc}  {status} {frame}  [{fmt_elapsed(now - start)}]"
            if len(line) > max_len:
                max_len = len(line)
            pad = " " * (max_len - len(line))
            sys.stdout.write("\r" + line + pad)
            sys.stdout.flush()
            time.sleep(0.1)
    finally:
        # Ensure we end on a newline in case of interruption
        sys.stdout.write("\n")
        sys.stdout.flush()


@dataclass
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: Optional[str] = None
    endpoint: Optional[str] = None

    @staticmethod
    def from_env() -> "R2Config":
        return R2Config(
            account_id=os.environ["R2_ACCOUNT_ID"],
            access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            bucket=os.environ.get("R2_BUCKET"),
        )

    def client(self):
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


def r2_list_mp3(r2: R2Config, bucket: str, prefix: str = "") -> Iterable[str]:
    s3 = r2.client()
    kwargs = {"Bucket": bucket, "Prefix": prefix} if prefix else {"Bucket": bucket}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".mp3"):
                yield key
        if resp.get("IsTruncated"):
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        else:
            break


def r2_presign_get(
    r2: R2Config, bucket: str, key: str, expires_seconds: int = 3600
) -> str:
    return r2.client().generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_seconds,
    )


def parse_bucket_key(s: str) -> Tuple[str, Optional[str]]:
    """
    Accepts:
      - "<bucket>/<key>"  -> returns (bucket, key)
      - "<bucket>"        -> returns (bucket, None)
    """
    if "/" in s:
        bucket, key = s.split("/", 1)
        if not bucket or not key:
            raise ValueError(f"Invalid bucket/key: {s}")
        return bucket, key
    return s, None


def transcribe_single_source(
    transcriber: aai.Transcriber,
    source: str,
    out_dir: Path,
    out_name_hint: Optional[str] = None,
) -> Path:
    submission = transcriber.submit(source)
    desc = Path(out_name_hint or source).name
    transcript = poll_with_progress(transcriber, submission.id, desc=desc)

    if transcript.status.name == "error":
        raise RuntimeError(transcript.error or "AssemblyAI returned error")

    data = {
        "id": transcript.json_response.get("id"),
        "status": transcript.json_response.get("status"),
        "error": transcript.json_response.get("error"),
        "confidence": transcript.json_response.get("confidence"),
        "audio_duration": transcript.json_response.get("audio_duration"),
        "text": transcript.json_response.get("text"),
        "words": transcript.json_response.get("words"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(desc).name + ".json")
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def transcribe_r2_bucket(
    r2: R2Config,
    transcriber: aai.Transcriber,
    bucket: str,
    prefix: str = "",
    out_dir: Path = Path("transcripts"),
    ttl: int = 3600,
    limit: Optional[int] = None,
) -> List[Path]:
    keys = list(r2_list_mp3(r2, bucket, prefix))
    if limit is not None:
        keys = keys[:limit]
    if not keys:
        return []

    outputs: List[Path] = []
    for key in keys:
        url = r2_presign_get(r2, bucket, key, ttl)
        out = transcribe_single_source(
            transcriber, url, out_dir, out_name_hint=Path(key).name
        )
        tqdm.write(f"✓ {bucket}/{key} → {out}")
        outputs.append(out)
    return outputs


def transcribe_r2_object(
    r2: R2Config,
    transcriber: aai.Transcriber,
    bucket: str,
    key: str,
    out_dir: Path = Path("transcripts"),
    ttl: int = 3600,
) -> Path:
    url = r2_presign_get(r2, bucket, key, ttl)
    return transcribe_single_source(
        transcriber, url, out_dir, out_name_hint=Path(key).name
    )
