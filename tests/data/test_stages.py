"""Tests for data pipeline stages."""

from typing import TYPE_CHECKING

import pytest
import torch

from demoodle.core.rng import RNG
from demoodle.core.types import Corpus, Dataset
from demoodle.data.stages import make_build_dataset_stage
from demoodle.ports.protocols import Stage
from demoodle.shell import runner
from demoodle.tokenizers.char import CharTokenizer

if TYPE_CHECKING:
    from pathlib import Path

    from demoodle.core.types import Artifact


@pytest.fixture
def tokenizer() -> CharTokenizer:
    chars = "abcdefghijklmnopqrstuvwxyz\n"
    return CharTokenizer(char_to_id={c: i for i, c in enumerate(chars)})


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(text="ab\ncd")


# ---------------------------------------------------------------------------
# Encoding correctness
# ---------------------------------------------------------------------------


def test_produces_dataset(tokenizer: CharTokenizer, corpus: Corpus) -> None:
    stage = make_build_dataset_stage()
    result = stage.run({"corpus": corpus, "tokenizer": tokenizer}, RNG(seed=0))
    assert isinstance(result["dataset"], Dataset)


def test_correct_token_sequence(tokenizer: CharTokenizer) -> None:
    corpus = Corpus(text="ab\ncd")
    stage = make_build_dataset_stage()
    result = stage.run({"corpus": corpus, "tokenizer": tokenizer}, RNG(seed=0))
    dataset = result["dataset"]
    assert isinstance(dataset, Dataset)
    expected = tokenizer.encode("ab\ncd\n")
    assert dataset.tokens.tolist() == expected


def test_tensor_dtype(tokenizer: CharTokenizer, corpus: Corpus) -> None:
    stage = make_build_dataset_stage()
    result = stage.run({"corpus": corpus, "tokenizer": tokenizer}, RNG(seed=0))
    dataset = result["dataset"]
    assert isinstance(dataset, Dataset)
    assert dataset.tokens.dtype == torch.long


def test_tensor_is_1d(tokenizer: CharTokenizer, corpus: Corpus) -> None:
    stage = make_build_dataset_stage()
    result = stage.run({"corpus": corpus, "tokenizer": tokenizer}, RNG(seed=0))
    dataset = result["dataset"]
    assert isinstance(dataset, Dataset)
    assert dataset.tokens.ndim == 1


def test_targets_are_inputs_shifted_by_one(tokenizer: CharTokenizer) -> None:
    corpus = Corpus(text="abc")
    stage = make_build_dataset_stage()
    result = stage.run({"corpus": corpus, "tokenizer": tokenizer}, RNG(seed=0))
    dataset = result["dataset"]
    assert isinstance(dataset, Dataset)
    t = dataset.tokens
    # For "abc\n": inputs = [a, b, c], targets = [b, c, \n]
    assert torch.equal(t[1:], torch.tensor(tokenizer.encode("bc\n"), dtype=torch.long))


# ---------------------------------------------------------------------------
# Trailing newline normalisation
# ---------------------------------------------------------------------------


def test_trailing_newline_added_when_missing(tokenizer: CharTokenizer) -> None:
    corpus = Corpus(text="abc")
    stage = make_build_dataset_stage()
    result = stage.run({"corpus": corpus, "tokenizer": tokenizer}, RNG(seed=0))
    dataset = result["dataset"]
    assert isinstance(dataset, Dataset)
    newline_id = tokenizer.char_to_id["\n"]
    assert dataset.tokens[-1].item() == newline_id


def test_trailing_newline_not_duplicated(tokenizer: CharTokenizer) -> None:
    corpus = Corpus(text="abc\n")
    stage = make_build_dataset_stage()
    result = stage.run({"corpus": corpus, "tokenizer": tokenizer}, RNG(seed=0))
    dataset = result["dataset"]
    assert isinstance(dataset, Dataset)
    newline_id = tokenizer.char_to_id["\n"]
    # last token is \n but second-to-last is not
    assert dataset.tokens[-1].item() == newline_id
    assert dataset.tokens[-2].item() != newline_id


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


def test_stage_runs_via_runner(
    tokenizer: CharTokenizer, corpus: Corpus, tmp_path: Path
) -> None:
    stage = make_build_dataset_stage()
    artifacts = runner.run(
        stages=[stage],
        initial_artifacts={"corpus": corpus, "tokenizer": tokenizer},
        rng=RNG(seed=0),
        cache_dir=tmp_path,
    )
    assert "dataset" in artifacts
    assert isinstance(artifacts["dataset"], Dataset)


def test_cache_hit_on_rerun(
    tokenizer: CharTokenizer, corpus: Corpus, tmp_path: Path
) -> None:
    calls: list[int] = []
    base_stage = make_build_dataset_stage()

    def counting_run(artifacts: dict[str, Artifact], rng: RNG) -> dict[str, Artifact]:
        calls.append(1)
        return base_stage.run(artifacts, rng)

    stage = Stage(
        name=base_stage.name,
        needs=base_stage.needs,
        produces=base_stage.produces,
        config_hash=base_stage.config_hash,
        run=counting_run,
    )
    initial: dict[str, Artifact] = {"corpus": corpus, "tokenizer": tokenizer}
    rng = RNG(seed=0)
    runner.run(stages=[stage], initial_artifacts=initial, rng=rng, cache_dir=tmp_path)
    runner.run(stages=[stage], initial_artifacts=initial, rng=rng, cache_dir=tmp_path)
    assert len(calls) == 1
