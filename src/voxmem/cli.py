from __future__ import annotations

import csv
import datetime as dt
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import humanize
import requests
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .csv_store import CsvRow, CsvStore
from .storage import StorageResult, TranscriptStorage
from .transcription import RateLimitEvent, TranscriptionError, Transcriber
from .transcribers.assemblyai import AssemblyAITranscriber
from .transcribers.elevenlabs import ElevenLabsTranscriber


console = Console()
app = typer.Typer(add_completion=False, rich_markup_mode="rich")


class TranscriptionProvider(str, Enum):
    elevenlabs = "elevenlabs"
    assemblyai = "assemblyai"


@dataclass(slots=True)
class WorkerResult:
    row_index: int
    filename: str
    transcription_id: str
    storage: StorageResult
    duration: float
    audio_duration: float | None


def configure_logging(log_target: Optional[str], output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("voxmem")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    if log_target in {"-", "--"}:
        handler: logging.Handler = logging.StreamHandler(sys.stdout)
    else:
        if log_target:
            log_path = Path(log_target)
        else:
            log_path = output_dir / "transcriptions.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path)
        console.print(f"[dim]Logging to {log_path}[/dim]")

    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def build_progress(total: int, skipped: int) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[green]done {task.fields[done]}"),
        TextColumn("[yellow]skip {task.fields[skip]}"),
        TextColumn("[red]err {task.fields[err]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def require_api_key(provider: TranscriptionProvider) -> str:
    if provider == TranscriptionProvider.elevenlabs:
        env_var = "ELEVENLABS_API_KEY"
    else:
        env_var = "ASSEMBLYAI_API_KEY"

    api_key = os.environ.get(env_var, "").strip()
    if not api_key:
        raise typer.BadParameter(
            f"{env_var} is not set; add it to the environment or .env"
        )
    return api_key


def handle_rate_limit(event: RateLimitEvent, logger: logging.Logger) -> None:
    reset = (
        event.reset_at.strftime("%H:%M:%S UTC")
        if event.reset_at is not None
        else "unknown"
    )
    message = (
        f"Rate limited on {event.endpoint}; sleeping {event.delay:.1f}s "
        f"(limit={event.limit or '?'} remaining={event.remaining or '?'}, reset {reset})"
    )
    console.print(f"[yellow]{message}[/yellow]")
    logger.warning(message)


def process_row(
    row: CsvRow,
    client_factory,
    storage: TranscriptStorage,
    logger: logging.Logger,
) -> WorkerResult:
    filename = row.filename
    if not filename:
        raise TranscriptionError("Row is missing filename")
    if not row.url:
        raise TranscriptionError(f"{filename}: missing URL")

    client = client_factory()
    start = time.monotonic()
    result = client.transcribe(row.url)
    transcription_id = result.transcription_id

    storage_result = storage.save_bundle(
        filename,
        transcription_id,
        result.payload,
    )
    audio_duration = _extract_audio_duration(result.payload)
    duration = time.monotonic() - start
    logger.info(
        "completed filename=%s transcription_id=%s folder=%s audio=%.1fs duration=%.1fs",
        filename,
        transcription_id,
        storage_result.folder,
        audio_duration or -1.0,
        duration,
    )
    return WorkerResult(
        row_index=row.index,
        filename=filename,
        transcription_id=transcription_id,
        storage=storage_result,
        duration=duration,
        audio_duration=audio_duration,
    )


def _extract_audio_duration(payload: Mapping[str, Any]) -> float | None:
    value = payload.get("audio_duration") if isinstance(payload, dict) else None
    try:
        if value is None:
            raise ValueError("no audio_duration")
        return float(value)
    except (TypeError, ValueError):
        words: list[Mapping[str, Any]] = []
        if isinstance(payload, dict):
            if "transcripts" in payload:
                for transcript in payload.get("transcripts") or []:
                    words.extend(transcript.get("words") or [])
            else:
                words.extend(payload.get("words") or [])
        ends = []
        for word in words:
            try:
                end = float(word.get("end"))
            except (TypeError, ValueError):
                continue
            ends.append(end)
        if not ends:
            return None
        max_end = max(ends)
        return max_end / 1000.0 if max_end > 10000 else max_end


@app.command()
def check(
    csv_path: Path = typer.Argument(..., exists=True, dir_okay=False, writable=True),
    offset: int = typer.Option(
        0, "--offset", min=0, help="Start processing from this row (0-based)"
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        min=1,
        help="Maximum number of rows to process (defaults to the rest of the file)",
    ),
):
    """Only check for duplicate filenames in the selected CSV slice."""
    load_dotenv()
    store = CsvStore(csv_path)
    selected = store.slice(offset=offset, limit=limit)
    if not selected:
        console.print("[yellow]No rows matched the requested offset/limit.[/yellow]")
        raise typer.Exit(code=0)

    try:
        store.ensure_unique_filenames(selected)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(
            f"[yellow]Checked {len(selected)} row(s); duplicates detected.[/yellow]"
        )
        raise typer.Exit(code=1)

    console.print(
        f"[green]Checked {len(selected)} row(s); no duplicates found.[/green]"
    )
    raise typer.Exit(code=0)


@app.command()
def export(
    csv_path: Path = typer.Argument(..., exists=True, dir_okay=False, writable=True),
    output: Path = typer.Option(
        Path("export.csv"),
        "--output",
        "-o",
        file_okay=True,
        dir_okay=False,
        help="Path to the export CSV (defaults to export.csv)",
    ),
    offset: int = typer.Option(
        0, "--offset", min=0, help="Start processing from this row (0-based)"
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        min=1,
        help="Maximum number of rows to process (defaults to the rest of the file)",
    ),
):
    """Export completed transcriptions to a CSV with JSON payload URLs."""
    load_dotenv()
    store = CsvStore(csv_path)
    selected = store.slice(offset=offset, limit=limit)
    if not selected:
        console.print("[yellow]No rows matched the requested offset/limit.[/yellow]")
        raise typer.Exit(code=0)

    completed = [row for row in selected if row.is_completed()]
    if not completed:
        console.print(
            "[yellow]No completed transcriptions found in selection.[/yellow]"
        )
        raise typer.Exit(code=0)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["file_id", "media_id", "filename", "transcription_url"],
        )
        writer.writeheader()
        for row in completed:
            stem = Path(row.filename).stem or row.filename
            writer.writerow(
                {
                    "file_id": row.data.get("file_id", "").strip(),
                    "media_id": row.data.get("media_id", "").strip(),
                    "filename": row.filename,
                    "transcription_url": (
                        f"https://files.pn.karmi.dev/{stem}/{row.transcription_id}.json"
                    ),
                }
            )

    console.print(
        f"[green]Exported {len(completed)} row(s) to {output} "
        f"(selected {len(selected)} total).[/green]"
    )


def _head_content_length(url: str, *, timeout: float, max_redirects: int) -> int | None:
    with requests.Session() as session:
        session.max_redirects = max_redirects
        resp = session.head(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={"User-Agent": "TranscriptionsStats/1.0"},
        )
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        return int(length) if length and length.isdigit() else None


@app.command()
def stats(
    csv_path: Path = typer.Argument(..., exists=True, dir_okay=False, writable=True),
    workers: int = typer.Option(
        10, "--workers", "-j", min=1, help="Number of worker threads"
    ),
    offset: int = typer.Option(
        0, "--offset", min=0, help="Start processing from this row (0-based)"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", min=1, help="Maximum number of rows to process"
    ),
    timeout: float = typer.Option(
        10.0, "--timeout", help="Per-request timeout in seconds"
    ),
    max_redirects: int = typer.Option(
        5, "--max-redirects", help="Maximum redirects to follow"
    ),
    bitrate: int = typer.Option(
        128_000,
        "--bitrate",
        help="Assumed bitrate in bits per second to estimate duration from bytes (default 128000 bps)",
    ),
    logfile: Path = typer.Option(
        Path("stats.log"),
        "--logfile",
        file_okay=True,
        dir_okay=False,
        help="File to write error details to (default: stats.log)",
    ),
) -> None:
    """Fetch HEAD Content-Length for URLs and report aggregate size and estimated duration."""
    load_dotenv()
    store = CsvStore(csv_path)
    selected = store.slice(offset=offset, limit=limit)
    targets = [(row.index, row.filename, row.url) for row in selected if row.url]

    if not targets:
        console.print("[yellow]No URLs to process in the selected slice.[/yellow]")
        raise typer.Exit(code=0)

    total = len(targets)
    console.print(f"Processing {total} URL(s) with {workers} worker(s)...")

    total_bytes = 0
    successes = failures = missing = 0
    error_messages: list[tuple[int, str, str, int | None]] = []
    status_counts: Counter[int] = Counter()

    def task(
        item: tuple[int, str, str],
    ) -> tuple[int, str, int | None, str | None, int | None]:
        idx, filename, url = item
        try:
            length = _head_content_length(
                url, timeout=timeout, max_redirects=max_redirects
            )
            return idx, filename, length, None, None
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            return idx, filename, None, str(exc), status
        except Exception as exc:
            return idx, filename, None, str(exc), None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, fname, length, err, status in executor.map(task, targets):
            if err is not None:
                failures += 1
                error_messages.append((idx, fname, err, status))
                if status is not None:
                    status_counts[status] += 1
            elif length is None:
                missing += 1
            else:
                successes += 1
                total_bytes += length

    total_size = humanize.naturalsize(total_bytes, binary=True)
    estimated_seconds = (total_bytes * 8 / bitrate) if bitrate > 0 else None
    if estimated_seconds is not None:
        hours = estimated_seconds / 3600
        estimated_duration = f"{humanize.intcomma(int(hours))} hours"
    else:
        estimated_duration = "n/a"

    console.print(
        "[bold]Stats:[/bold] "
        f"urls={total} ok={successes} missing_length={missing} errors={failures} "
        f"size={total_size} est_duration={estimated_duration}"
    )
    if failures and error_messages:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("w", encoding="utf-8") as fh:
            for idx, fname, msg, status in error_messages:
                fh.write(f"row={idx} file={fname} status={status or '?'} error={msg}\n")
        if status_counts:
            summary = ", ".join(
                f"[yellow not bold]{code}[/]={count}"
                for code, count in sorted(status_counts.items())
            )
            console.print(f"[bold]Error:[/bold] {summary}")


@app.command()
def transcribe(
    csv_path: Path = typer.Argument(..., exists=True, dir_okay=False, writable=True),
    output: Path = typer.Option(
        Path("transcripts"),
        "--output",
        "-o",
        file_okay=False,
        dir_okay=True,
        help="Directory where transcript artifacts will be stored",
    ),
    workers: int = typer.Option(
        5, "--workers", "-j", min=1, help="Number of worker threads"
    ),
    offset: int = typer.Option(
        0, "--offset", min=0, help="Start processing from this row (0-based)"
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        min=1,
        help="Maximum number of rows to process (defaults to the rest of the file)",
    ),
    timeout: int = typer.Option(
        3600,
        "--timeout",
        help="Maximum seconds to wait for a single transcription",
    ),
    poll_interval: float = typer.Option(
        2.0,
        "--poll-interval",
        help="Seconds to wait between polling providers",
    ),
    provider: TranscriptionProvider = typer.Option(
        TranscriptionProvider.elevenlabs,
        "--provider",
        help="Transcription provider to use (default: elevenlabs)",
    ),
    logfile: Optional[str] = typer.Option(
        None,
        "--logfile",
        help="Path to log file (use '--' for stdout, default writes to OUTPUT/transcriptions.log)",
    ),
):
    """Transcribe audio files listed in CSV via the selected provider."""

    load_dotenv()
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(logfile, output)

    api_key = require_api_key(provider)
    store = CsvStore(csv_path)
    selected = store.slice(offset=offset, limit=limit)
    if not selected:
        console.print("[yellow]No rows matched the requested offset/limit.[/yellow]")
        raise typer.Exit(code=0)

    store.ensure_unique_filenames(selected)
    pending = store.pending(selected)
    skipped = len(selected) - len(pending)

    if not pending:
        console.print(
            "[green]All selected rows already have completed transcriptions.[/green]"
        )
        raise typer.Exit(code=0)

    storage = TranscriptStorage(output)
    rate_limit_callback = lambda event: handle_rate_limit(event, logger)
    thread_local: threading.local = threading.local()

    def client_factory() -> Transcriber:
        client = getattr(thread_local, "client", None)
        if client is None:
            if provider == TranscriptionProvider.elevenlabs:
                client = ElevenLabsTranscriber(
                    api_key=api_key,
                    request_timeout=float(timeout),
                    timeout=float(timeout),
                    poll_interval=poll_interval,
                    on_rate_limit=rate_limit_callback,
                )
            else:
                client = AssemblyAITranscriber(
                    api_key=api_key,
                    poll_interval=poll_interval,
                    timeout=float(timeout),
                    on_rate_limit=rate_limit_callback,
                )
            thread_local.client = client
        return client

    console.print(
        f"Processing {len(pending)} row(s) (skipping {skipped}) from {csv_path} "
        f"using {provider.value}"
    )
    logger.info(
        "start provider=%s csv=%s rows=%d skipped=%d workers=%d offset=%d limit=%s",
        provider.value,
        csv_path,
        len(pending),
        skipped,
        workers,
        offset,
        limit,
    )

    progress = build_progress(total=len(pending), skipped=skipped)
    task_id = progress.add_task(
        "transcribing",
        total=len(pending),
        done=0,
        skip=skipped,
        err=0,
    )

    completed = 0
    failed = 0

    start_time = time.monotonic()

    with progress:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(process_row, row, client_factory, storage, logger): row
                for row in pending
            }

            for future in as_completed(future_map):
                row = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - runtime errors
                    failed += 1
                    state = _status_label("error", "red")
                    filename = row.filename or f"row #{row.index}"
                    exc_name = type(exc).__name__
                    detail = _exc_summary(exc)
                    console.print(f"{state} {filename} [dim]{exc_name}: {detail}[/dim]")
                    logger.error(
                        "failed filename=%s error=%s location=%s detail=%s",
                        filename,
                        exc,
                        _last_trace_location(exc),
                        detail,
                    )
                    store.mark_failed(row.index, str(exc))
                    progress.update(task_id, advance=1, err=failed)
                    continue

                store.mark_completed(result.row_index, result.transcription_id)
                completed += 1
                progress.update(task_id, advance=1, done=completed)
                state = _status_label("done", "green")
                detail = (
                    f"• {format_duration(result.audio_duration)} in "
                    f"{format_duration(result.duration)}"
                )
                console.print(f"{state} {result.filename} [dim]{detail}[/dim]")

    total_duration = time.monotonic() - start_time
    summary = (
        f"completed={completed} skipped={skipped} failed={failed} output={storage.root}"
    )
    summary_full = f"{summary} duration={format_duration(total_duration)}"
    logger.info(summary_full)
    console.print(f"[bold]Finished:[/bold] {summary_full}")

    if failed:
        raise typer.Exit(code=1)


def main() -> None:
    app()


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "n/a"
    if value < 0:
        return "n/a"
    delta = dt.timedelta(seconds=value)
    return humanize.naturaldelta(delta)


def _status_label(label: str, color: str, width: int = 6) -> str:
    padded = f"{label:<{width}}"
    return f"[{color}]{padded}[/{color}]"


def _last_trace_location(exc: BaseException) -> str:
    tb = exc.__traceback__
    if tb is None:
        return "unknown"
    frames = traceback.extract_tb(tb)
    if not frames:
        return "unknown"
    last = frames[-1]
    return f"{last.filename}:{last.lineno}"


def _exc_summary(exc: BaseException, limit: int = 60) -> str:
    text = str(exc).replace("\n", " ").strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text or "(no detail)"
