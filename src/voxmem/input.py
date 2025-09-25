from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, Tuple

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
