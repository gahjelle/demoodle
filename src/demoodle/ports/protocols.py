"""Behavioral protocols and the Stage dataclass — the pipeline's swap points."""

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

    def init_state(self, rng: RNG) -> Policy:
        """Return a freshly initialized Policy. Config is bound at construction."""
        ...

    def forward(self, policy: Policy, tokens: Seq) -> Output:
        """Run a forward pass and return logits."""
        ...


class InspectableProtocol(Protocol):
    """Sample next tokens and optionally expose internals."""

    def call(
        self,
        seq: Seq,
        policy: Policy,
        rng: RNG,
        temperature: float,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> Output:
        """Return logits and sampled next token id for the given context."""
        ...

    def explain(self, seq: Seq, policy: Policy) -> dict[str, Any]:  # noqa: ARG002
        """Return interpretability data for the given context; default is empty."""
        return {}


@dataclass(frozen=True)
class Stage:
    """Immutable pipeline stage: declares its artifact dependencies and run logic."""

    name: str
    needs: list[str]
    produces: list[str]
    config_hash: str
    run: Callable[[dict[str, Artifact], RNG], dict[str, Artifact]]
