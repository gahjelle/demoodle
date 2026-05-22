"""Tests for CharTokenizer and make_train_tokenizer_stage."""

from dataclasses import FrozenInstanceError

import pytest

from demoodle.core.rng import RNG
from demoodle.core.types import Corpus
from demoodle.tokenizers.char import CharTokenizer, make_train_tokenizer_stage


@pytest.fixture
def tokenizer() -> CharTokenizer:
    chars = "abcdefghijklmnopqrstuvwxyz\n"
    return CharTokenizer(char_to_id={c: i for i, c in enumerate(chars)})


def test_round_trip(tokenizer: CharTokenizer) -> None:
    s = "hello\nworld\n"
    assert tokenizer.decode(tokenizer.encode(s)) == s


def test_vocab_size(tokenizer: CharTokenizer) -> None:
    assert tokenizer.vocab_size == len(tokenizer.char_to_id)


def test_encode_unknown_char_raises(tokenizer: CharTokenizer) -> None:
    with pytest.raises(KeyError):
        tokenizer.encode("hello!")


def test_frozen(tokenizer: CharTokenizer) -> None:
    with pytest.raises(FrozenInstanceError):
        tokenizer.char_to_id = {}  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Stage tests
# ---------------------------------------------------------------------------


def test_stage_produces_char_tokenizer() -> None:
    corpus = Corpus(text="emma\nolivia\nava\n")
    stage = make_train_tokenizer_stage()
    result = stage.run({"corpus": corpus}, RNG(seed=0))
    assert isinstance(result["tokenizer"], CharTokenizer)


def test_stage_vocab_covers_all_corpus_chars() -> None:
    corpus = Corpus(text="emma\nolivia\nava\n")
    stage = make_train_tokenizer_stage()
    result = stage.run({"corpus": corpus}, RNG(seed=0))
    tokenizer = result["tokenizer"]
    assert isinstance(tokenizer, CharTokenizer)
    for char in corpus.text:
        assert char in tokenizer.char_to_id


def test_stage_includes_newline_as_regular_vocab_entry() -> None:
    corpus = Corpus(text="emma\nava\n")
    stage = make_train_tokenizer_stage()
    result = stage.run({"corpus": corpus}, RNG(seed=0))
    tokenizer = result["tokenizer"]
    assert isinstance(tokenizer, CharTokenizer)
    assert "\n" in tokenizer.char_to_id
