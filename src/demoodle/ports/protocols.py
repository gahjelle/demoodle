"""Behavioral protocols and the Stage dataclass — the pipeline's swap points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from demoodle.core.rng import RNG
    from demoodle.core.types import Artifact, Output, Policy, Seq


class TokenizerProtocol(Protocol):
    """Encode text to token ids and decode back."""

    vocab_size: int

    def encode(self, text: str) -> list[int]:
        """Return token ids for the given text."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Return text for the given token ids."""
        ...


class ArchitectureProtocol(Protocol):
    """Initialize model state and run forward passes."""

    def init_state(self) -> Policy:
        """Return a freshly initialized Policy. Signature extended by W10."""
        ...

    def forward(self, policy: Policy, tokens: Seq) -> Output:
        """Run a forward pass and return logits."""
        ...


class InspectableProtocol(Protocol):
    """Sample next tokens and optionally expose internals."""

    def call(self, seq: Seq, temperature: float) -> int:
        """Sample the next token id given a context sequence."""
        ...

    def explain(self) -> dict[str, Any]:
        """Return interpretability data; default is empty (no inspection)."""
        return {}


@dataclass(frozen=True)
class Stage:
    """Immutable pipeline stage: declares its artifact dependencies and run logic."""

    name: str
    needs: list[str]
    produces: list[str]
    config_hash: str
    run: Callable[[dict[str, Artifact], RNG], dict[str, Artifact]]
