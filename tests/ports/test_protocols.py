import dataclasses
from typing import Any

import pytest
import torch
from torch import nn

from demoodle.core.rng import RNG
from demoodle.core.types import Artifact, Corpus, Output, Policy, Seq
from demoodle.ports import (
    ArchitectureProtocol,
    InspectableProtocol,
    Stage,
    TokenizerProtocol,
)

# ---------------------------------------------------------------------------
# TokenizerProtocol
# ---------------------------------------------------------------------------


class _DummyTokenizer:
    vocab_size: int = 10

    def encode(self, text: str) -> list[int]:
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i + 97) for i in ids)


def test_tokenizer_protocol_dummy_satisfies() -> None:
    tok: TokenizerProtocol = _DummyTokenizer()
    assert tok.vocab_size == 10


def test_tokenizer_protocol_round_trip() -> None:
    class _RoundTrip:
        vocab_size: int = 256

        def encode(self, text: str) -> list[int]:
            return list(text.encode())

        def decode(self, ids: list[int]) -> str:
            return bytes(ids).decode()

    tok = _RoundTrip()
    assert tok.decode(tok.encode("hello")) == "hello"


# ---------------------------------------------------------------------------
# ArchitectureProtocol
# ---------------------------------------------------------------------------


class _DummyArchitecture:
    def init_state(self) -> Policy:
        return Policy(model=nn.Linear(4, 4))

    def forward(self, policy: Policy, tokens: Seq) -> Output:  # noqa: ARG002
        return Output(logits=torch.zeros(4))


def test_architecture_protocol_dummy_satisfies() -> None:
    arch: ArchitectureProtocol = _DummyArchitecture()
    policy = arch.init_state()
    assert isinstance(policy, Policy)


# ---------------------------------------------------------------------------
# InspectableProtocol
# ---------------------------------------------------------------------------


class _MinimalInspectable(InspectableProtocol):
    def call(self, seq: Seq, temperature: float) -> int:  # noqa: ARG002
        return 0


class _FullInspectable:
    def call(self, seq: Seq, temperature: float) -> int:  # noqa: ARG002
        return 1

    def explain(self) -> dict[str, Any]:
        return {"attention": [[0.5, 0.5]]}


def test_inspectable_protocol_minimal_satisfies() -> None:
    obj: InspectableProtocol = _MinimalInspectable()
    assert obj.call(torch.zeros(1, dtype=torch.long), 1.0) == 0


def test_inspectable_protocol_explain_default_returns_empty_dict() -> None:
    obj: InspectableProtocol = _MinimalInspectable()
    assert obj.explain() == {}


def test_inspectable_protocol_explain_override_returns_custom_data() -> None:
    obj: InspectableProtocol = _FullInspectable()
    result = obj.explain()
    assert "attention" in result


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def _noop_run(artifacts: dict[str, Artifact], _rng: RNG) -> dict[str, Artifact]:
    return artifacts


def test_stage_is_frozen() -> None:
    stage = Stage(name="test", needs=[], produces=[], run=_noop_run)
    with pytest.raises(dataclasses.FrozenInstanceError):
        stage.name = "other"  # ty: ignore[invalid-assignment]


def test_stage_run_pure_under_same_rng() -> None:
    inputs: dict[str, Artifact] = {"corpus": Corpus(text="hello")}
    rng = RNG(seed=42)
    stage = Stage(name="identity", needs=["corpus"], produces=["corpus"], run=_noop_run)
    assert stage.run(inputs, rng) == stage.run(inputs, rng)


# ---------------------------------------------------------------------------
# Importable from demoodle.ports
# ---------------------------------------------------------------------------


def test_all_names_importable_from_ports() -> None:
    from demoodle.ports import (  # noqa: PLC0415
        ArchitectureProtocol,
        InspectableProtocol,
        Stage,
        TokenizerProtocol,
    )

    assert TokenizerProtocol is not None
    assert ArchitectureProtocol is not None
    assert InspectableProtocol is not None
    assert Stage is not None
