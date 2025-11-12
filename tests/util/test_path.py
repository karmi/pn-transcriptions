from __future__ import annotations

import pytest

from voxmem.util.path import normalize_to_dirname


def test_basic_name() -> None:
    assert normalize_to_dirname("audio.mp3") == "audio"


def test_strips_directories() -> None:
    assert normalize_to_dirname("../weird/audio.mp3") == "audio"


def test_unicode_and_spaces() -> None:
    assert normalize_to_dirname("Žluťoučký kůň.mp3") == "Zlutoucky_kun"


def test_symbols_are_replaced() -> None:
    assert normalize_to_dirname("foo*?bar.mp3") == "foo_bar"


def test_trim_to_max_length() -> None:
    name = "a" * 200
    assert normalize_to_dirname(name) == "a" * 100


def test_reject_missing_name() -> None:
    with pytest.raises(ValueError):
        normalize_to_dirname("")


def test_reject_unusable_name() -> None:
    with pytest.raises(ValueError):
        normalize_to_dirname("!!!")
