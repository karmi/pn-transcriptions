from __future__ import annotations

import pytest

from voxmem.csv_store import CsvRow, CsvStore


class DummyStore(CsvStore):
    def __init__(self) -> None:
        # bypass file loading
        self.path = None  # type: ignore[assignment]
        self.fieldnames = []
        self.rows = []
        self._lock = None  # type: ignore[assignment]


def make_row(idx: int, filename: str) -> CsvRow:
    return CsvRow(index=idx, data={"filename": filename, "url": "http://example.com"})


def test_duplicate_error_lists_rows() -> None:
    store = DummyStore()
    rows = [make_row(1, "dup.mp3"), make_row(3, "dup.mp3")]
    with pytest.raises(ValueError) as excinfo:
        store.ensure_unique_filenames(rows)
    message = str(excinfo.value)
    assert "Duplicate filenames detected" in message
    assert "Row 1: dup.mp3" in message
    assert "Row 3: dup.mp3" in message
