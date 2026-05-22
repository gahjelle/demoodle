"""Tests for corpus loaders."""

import pytest

from demoodle.config import config
from demoodle.core.types import Corpus
from demoodle.data.loaders import load_corpus


def test_load_names_returns_corpus() -> None:
    corpus = load_corpus("names")
    assert isinstance(corpus, Corpus)


def test_load_names_is_nonempty() -> None:
    corpus = load_corpus("names")
    assert corpus.text


def test_load_names_line_count() -> None:
    corpus = load_corpus("names")
    lines = corpus.text.splitlines()
    assert 30_000 < len(lines) < 35_000


def test_load_names_no_trailing_whitespace() -> None:
    corpus = load_corpus("names")
    assert not corpus.text.endswith("\n")


def test_load_unknown_corpus_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus("nonexistent")


def test_load_corpus_uses_active_config() -> None:
    corpus = load_corpus(config.corpus.active)
    assert isinstance(corpus, Corpus)
    assert corpus.text
