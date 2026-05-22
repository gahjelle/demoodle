"""Tests for shell/persistence.py -- content-addressed artifact cache."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch
from torch import nn

from demoodle.core.rng import RNG
from demoodle.core.types import Artifact, Corpus, Dataset, Metrics, Policy, Tokenizer
from demoodle.ports import Stage
from demoodle.shell.persistence import _hash_artifact, cache_key, load, save

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop_run(artifacts: dict[str, Artifact], _rng: RNG) -> dict[str, Artifact]:
    return artifacts


def _stage(name: str = "s", config_hash: str = "") -> Stage:
    return Stage(
        name=name, needs=[], produces=[], config_hash=config_hash, run=_noop_run
    )


def _rng(seed: int = 0) -> RNG:
    return RNG(seed=seed)


# ---------------------------------------------------------------------------
# 2.1 / 2.3 - Artifact hashing
# ---------------------------------------------------------------------------


def test_hash_corpus_is_stable() -> None:
    c = Corpus(text="hello")
    assert _hash_artifact(c) == _hash_artifact(c)
    assert isinstance(_hash_artifact(c), str)
    assert len(_hash_artifact(c)) == 64  # sha256 hex


def test_hash_corpus_differs_on_different_text() -> None:
    assert _hash_artifact(Corpus(text="hello")) != _hash_artifact(Corpus(text="world"))


def test_hash_metrics_stable_and_sensitive() -> None:
    m = Metrics(losses=[0.1, 0.2])
    assert _hash_artifact(m) == _hash_artifact(m)
    assert _hash_artifact(Metrics(losses=[0.1, 0.2])) != _hash_artifact(
        Metrics(losses=[0.1, 0.3])
    )


def test_hash_dataset_stable_and_sensitive() -> None:
    d1 = Dataset(tokens=torch.tensor([1, 2, 3]))
    d2 = Dataset(tokens=torch.tensor([1, 2, 4]))
    assert _hash_artifact(d1) == _hash_artifact(d1)
    assert _hash_artifact(d1) != _hash_artifact(d2)


def test_hash_tokenizer_stable_and_sensitive() -> None:
    t1 = Tokenizer(vocab_size=10)
    t2 = Tokenizer(vocab_size=20)
    assert _hash_artifact(t1) == _hash_artifact(t1)
    assert _hash_artifact(t1) != _hash_artifact(t2)


def test_hash_policy_stable_and_sensitive() -> None:
    model_a = nn.Linear(2, 2)
    model_b = nn.Linear(2, 2)
    # two randomly-initialised linear layers are almost certainly different
    p1 = Policy(model=model_a)
    p2 = Policy(model=model_b)
    assert _hash_artifact(p1) == _hash_artifact(p1)
    assert _hash_artifact(p1) != _hash_artifact(p2)


# ---------------------------------------------------------------------------
# 3.1-3.4 - cache_key
# ---------------------------------------------------------------------------


def test_cache_key_same_inputs_same_key() -> None:
    stage = _stage()
    inputs: dict[str, Artifact] = {"c": Corpus(text="x")}
    rng = _rng(1)
    assert cache_key(stage, inputs, rng) == cache_key(stage, inputs, rng)


def test_cache_key_different_seed_different_key() -> None:
    stage = _stage()
    inputs: dict[str, Artifact] = {}
    assert cache_key(stage, inputs, _rng(1)) != cache_key(stage, inputs, _rng(2))


def test_cache_key_different_config_hash_different_key() -> None:
    inputs: dict[str, Artifact] = {}
    rng = _rng()
    assert cache_key(_stage(config_hash="aaa"), inputs, rng) != cache_key(
        _stage(config_hash="bbb"), inputs, rng
    )


def test_cache_key_different_artifact_content_different_key() -> None:
    stage = _stage()
    rng = _rng()
    inputs_a: dict[str, Artifact] = {"c": Corpus(text="hello")}
    inputs_b: dict[str, Artifact] = {"c": Corpus(text="world")}
    assert cache_key(stage, inputs_a, rng) != cache_key(stage, inputs_b, rng)


# ---------------------------------------------------------------------------
# 3.7 - git fallback
# ---------------------------------------------------------------------------


def test_git_id_falls_back_on_missing_binary() -> None:
    from demoodle.shell import persistence  # noqa: PLC0415

    with patch("subprocess.run", side_effect=FileNotFoundError):
        git_id = persistence._git_id()
    assert git_id == ""


def test_git_id_falls_back_in_non_repo() -> None:
    from demoodle.shell import persistence  # noqa: PLC0415

    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(128, "git")):
        git_id = persistence._git_id()
    assert git_id == ""


def test_cache_key_returns_string_when_no_git() -> None:
    stage = _stage()
    inputs: dict[str, Artifact] = {}
    rng = _rng()
    with patch("demoodle.shell.persistence._GIT_ID", ""):
        key = cache_key(stage, inputs, rng)
    assert isinstance(key, str)
    assert len(key) == 64


# ---------------------------------------------------------------------------
# 4.1-4.4 - save / load
# ---------------------------------------------------------------------------


def test_save_load_corpus_roundtrip(tmp_path: Path) -> None:
    original = Corpus(text="the quick brown fox")
    key = "testkey"
    save(key, {"c": original}, tmp_path)
    loaded = load(key, tmp_path)
    assert loaded is not None
    assert loaded["c"] == original


def test_save_load_dataset_roundtrip(tmp_path: Path) -> None:
    tokens = torch.tensor([1, 2, 3, 4, 5])
    original = Dataset(tokens=tokens)
    key = "dskey"
    save(key, {"d": original}, tmp_path)
    loaded = load(key, tmp_path)
    assert loaded is not None
    assert torch.equal(loaded["d"].tokens, tokens)  # ty: ignore[unresolved-attribute]


def test_save_load_policy_roundtrip(tmp_path: Path) -> None:
    model = nn.Linear(4, 4)
    original = Policy(model=model)
    key = "policykey"
    save(key, {"p": original}, tmp_path)
    loaded = load(key, tmp_path)
    assert loaded is not None
    loaded_policy: Policy = loaded["p"]  # ty: ignore[invalid-assignment]
    for k in model.state_dict():
        assert torch.equal(loaded_policy.model.state_dict()[k], model.state_dict()[k])


def test_load_missing_key_returns_none(tmp_path: Path) -> None:
    assert load("doesnotexist", tmp_path) is None
