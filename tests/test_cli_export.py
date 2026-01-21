from __future__ import annotations

import csv
from pathlib import Path

from typer.testing import CliRunner

from voxmem.cli import app


runner = CliRunner()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "file_id",
        "media_id",
        "filename",
        "url",
        "transcription_id",
        "status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_export_writes_completed_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        [
            {
                "file_id": "1",
                "media_id": "10",
                "filename": "one.mp3",
                "url": "http://example.com/1",
                "transcription_id": "abc",
                "status": "completed",
                "error": "",
            },
            {
                "file_id": "2",
                "media_id": "20",
                "filename": "two.mp3",
                "url": "http://example.com/2",
                "transcription_id": "",
                "status": "",
                "error": "",
            },
        ],
    )

    output = tmp_path / "export.csv"
    result = runner.invoke(app, ["export", str(csv_path), "--output", str(output)])
    assert result.exit_code == 0, result.output

    with output.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 1
    row = rows[0]
    assert row["file_id"] == "1"
    assert row["media_id"] == "10"
    assert row["filename"] == "one.mp3"
    assert row["transcription_url"] == "https://files.pn.karmi.dev/one/abc.json"
