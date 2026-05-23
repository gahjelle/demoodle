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
from demoodle.tokenizers.char import CharTokenizer

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


def test_char_tokenizer_satisfies_tokenizer_protocol() -> None:
    tok: TokenizerProtocol = CharTokenizer(char_to_id={"a": 0, "b": 1, "\n": 2})
    assert tok.vocab_size == 3
    assert tok.encode("ab") == [0, 1]
    assert tok.decode([2, 0]) == "\na"


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
    def init_state(self, rng: RNG) -> Policy:  # noqa: ARG002
        return Policy(model=nn.Linear(4, 4))

    def forward(self, policy: Policy, tokens: Seq) -> Output:  # noqa: ARG002
        return Output(logits=torch.zeros(4))


def test_architecture_protocol_dummy_satisfies() -> None:
    arch: ArchitectureProtocol = _DummyArchitecture()
    policy = arch.init_state(RNG(seed=0))
    assert isinstance(policy, Policy)


# ---------------------------------------------------------------------------
# InspectableProtocol
# ---------------------------------------------------------------------------


class _MinimalInspectable(InspectableProtocol):
    def call(
        self,
        seq: Seq,  # noqa: ARG002
        policy: Policy,  # noqa: ARG002
        rng: RNG,  # noqa: ARG002
        temperature: float,  # noqa: ARG002
        top_k: int | None = None,  # noqa: ARG002
        top_p: float | None = None,  # noqa: ARG002
    ) -> Output:
        return Output(
            logits=torch.zeros(4), sampled_ids=torch.zeros(1, dtype=torch.long)
        )


class _FullInspectable:
    def call(
        self,
        seq: Seq,  # noqa: ARG002
        policy: Policy,  # noqa: ARG002
        rng: RNG,  # noqa: ARG002
        temperature: float,  # noqa: ARG002
        top_k: int | None = None,  # noqa: ARG002
        top_p: float | None = None,  # noqa: ARG002
    ) -> Output:
        return Output(
            logits=torch.zeros(4), sampled_ids=torch.zeros(1, dtype=torch.long)
        )

    def explain(self, seq: Seq, policy: Policy) -> dict[str, Any]:  # noqa: ARG002
        return {"attention": [[0.5, 0.5]]}


def test_inspectable_protocol_minimal_satisfies() -> None:
    dummy_policy = Policy(model=nn.Linear(4, 4))
    obj: InspectableProtocol = _MinimalInspectable()
    result = obj.call(torch.zeros(1, dtype=torch.long), dummy_policy, RNG(seed=0), 1.0)
    assert result.sampled_ids is not None


def test_inspectable_protocol_explain_default_returns_empty_dict() -> None:
    dummy_policy = Policy(model=nn.Linear(4, 4))
    obj: InspectableProtocol = _MinimalInspectable()
    assert obj.explain(torch.zeros(1, dtype=torch.long), dummy_policy) == {}


def test_inspectable_protocol_explain_override_returns_custom_data() -> None:
    dummy_policy = Policy(model=nn.Linear(4, 4))
    obj: InspectableProtocol = _FullInspectable()
    result = obj.explain(torch.zeros(1, dtype=torch.long), dummy_policy)
    assert "attention" in result


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def _noop_run(artifacts: dict[str, Artifact], _rng: RNG) -> dict[str, Artifact]:
    return artifacts


def test_stage_is_frozen() -> None:
    stage = Stage(name="test", needs=[], produces=[], config_hash="", run=_noop_run)
    with pytest.raises(dataclasses.FrozenInstanceError):
        stage.name = "other"  # ty: ignore[invalid-assignment]


def test_stage_run_pure_under_same_rng() -> None:
    inputs: dict[str, Artifact] = {"corpus": Corpus(text="hello")}
    rng = RNG(seed=42)
    stage = Stage(
        name="identity",
        needs=["corpus"],
        produces=["corpus"],
        config_hash="",
        run=_noop_run,
    )
    assert stage.run(inputs, rng) == stage.run(inputs, rng)


def test_stage_construction_requires_config_hash() -> None:
    with pytest.raises(TypeError):
        Stage(name="test", needs=[], produces=[], run=_noop_run)  # ty: ignore[missing-argument]


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
