import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import click
import assemblyai as aai

from lib.transcription import (
    R2Config,
    init_assemblyai,
    parse_bucket_key,
    transcribe_single_source,
    transcribe_r2_bucket,
    transcribe_r2_object,
)


@click.group()
def cli():
    """Transcribe MP3s from Cloudflare R2 using AssemblyAI"""
    pass


@cli.command("file")
@click.argument("path_or_r2", type=str)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("transcripts"),
)
@click.option(
    "--poll", type=float, default=2.0, show_default=True, help="Polling interval (s)"
)
def file_cmd(path_or_r2: str, out_dir: Path, poll: float):
    """Transcribe a single local file OR R2 object '<bucket>/<key>'."""
    transcriber: aai.Transcriber = init_assemblyai(polling_interval=poll)

    if os.path.exists(path_or_r2):
        out = transcribe_single_source(transcriber, path_or_r2, out_dir)
        click.echo(f"✓ {path_or_r2} → {out}")
        return

    bucket, key = parse_bucket_key(path_or_r2)
    r2 = R2Config.from_env()
    out = transcribe_r2_object(r2, transcriber, bucket, key, out_dir=out_dir)
    click.echo(f"✓ {bucket}/{key} → {out}")


@cli.command("bucket")
@click.argument("bucket_or_prefix", type=str)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("transcripts"),
)
@click.option("--limit", type=int, default=None)
@click.option(
    "--ttl", type=int, default=3600, show_default=True, help="Presigned URL TTL (s)"
)
@click.option(
    "--poll", type=float, default=2.0, show_default=True, help="Polling interval (s)"
)
def bucket_cmd(
    bucket_or_prefix: str,
    out_dir: Path,
    limit: int,
    ttl: int,
    poll: float,
):
    """Transcribe all .mp3 in a bucket (optionally under a prefix). Accepts 'bucket' or 'bucket/prefix'."""
    transcriber = init_assemblyai(polling_interval=poll)
    r2 = R2Config.from_env()
    if "/" in bucket_or_prefix:
        bucket, prefix = bucket_or_prefix.split("/", 1)
    else:
        bucket, prefix = bucket_or_prefix, ""
    outs = transcribe_r2_bucket(
        r2, transcriber, bucket, prefix, out_dir=out_dir, ttl=ttl, limit=limit
    )
    if not outs:
        click.echo("No .mp3 objects found.", err=True)
    else:
        click.echo(f"Done. Wrote {len(outs)} transcripts → {out_dir}")


if __name__ == "__main__":
    cli()
