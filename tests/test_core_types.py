import dataclasses
from typing import get_args

import pytest
import torch
from torch import nn

from demoodle.core.types import (
    Artifact,
    Corpus,
    Dataset,
    Output,
    Policy,
    TrainingMetrics,
)
from demoodle.tokenizers.char import CharTokenizer

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_output_defaults_sampled_ids_to_none() -> None:
    out = Output(logits=torch.zeros(5))
    assert out.sampled_ids is None


def test_output_is_frozen() -> None:
    out = Output(logits=torch.zeros(5))
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.logits = torch.ones(5)  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def test_corpus_holds_text() -> None:
    c = Corpus(text="alice\nbob\n")
    assert c.text == "alice\nbob\n"


def test_corpus_is_frozen() -> None:
    c = Corpus(text="hello")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.text = "world"  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_dataset_holds_tokens() -> None:
    tokens = torch.arange(10)
    ds = Dataset(tokens=tokens)
    assert ds.tokens.shape == (10,)


def test_dataset_is_frozen() -> None:
    ds = Dataset(tokens=torch.zeros(5))
    with pytest.raises(dataclasses.FrozenInstanceError):
        ds.tokens = torch.ones(5)  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_policy_defaults_value_head_to_none() -> None:
    model = nn.Linear(4, 4)
    p = Policy(model=model)
    assert p.value_head is None


def test_policy_accepts_value_head() -> None:
    model = nn.Linear(4, 4)
    head = nn.Linear(4, 1)
    p = Policy(model=model, value_head=head)
    assert p.value_head is head


def test_policy_is_frozen() -> None:
    p = Policy(model=nn.Linear(4, 4))
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.model = nn.Linear(4, 4)  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# TrainingMetrics
# ---------------------------------------------------------------------------


def test_metrics_holds_losses() -> None:
    m = TrainingMetrics(losses=[1.0, 0.8, 0.6])
    assert m.losses == [1.0, 0.8, 0.6]


def test_metrics_is_frozen() -> None:
    m = TrainingMetrics(losses=[1.0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.losses = [0.5]  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Artifact union
# ---------------------------------------------------------------------------


def test_artifact_union_includes_all_variants() -> None:
    variants = set(get_args(Artifact.__value__))
    assert variants == {CharTokenizer, Corpus, Dataset, Policy, TrainingMetrics}
