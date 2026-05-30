"""Trigram architecture: a VxVxV weight tensor predicting from the last two tokens."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from demoodle.architectures.sampling import sample
from demoodle.core.types import Output, Policy, Seq

if TYPE_CHECKING:
    from demoodle.core.rng import RNG


class TrigramModel(nn.Module):
    """VxVxV weight tensor: table[t_prev, t_cur] gives the next-token distribution."""

    def __init__(self, vocab_size: int) -> None:
        """Initialise with a zero-filled (V, V, V) weight tensor."""
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(vocab_size, vocab_size, vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for the next token.

        x: shape (2,) for a single context pair, or (batch, 2) for batched training.
        """
        if x.ndim == 1:
            return self.weight[x[0], x[1]]
        return self.weight[x[:, 0], x[:, 1]]


@dataclass(frozen=True)
class TrigramArchitecture:
    """VxVxV weight-tensor model. Stateless: no Policy held internally."""

    vocab_size: int
    context_length: int = 2

    def init_state(self, rng: RNG) -> Policy:
        """Return a freshly initialised Policy. Pure function of rng."""
        model = TrigramModel(self.vocab_size)
        nn.init.normal_(model.weight, generator=rng.generator())
        return Policy(model=model)

    def forward(self, policy: Policy, tokens: Seq) -> Output:
        """Run a forward pass using the last two tokens."""
        logits: torch.Tensor = policy.model(tokens[-2:])
        return Output(logits=logits)

    def call(
        self,
        seq: Seq,
        policy: Policy,
        rng: RNG,
        temperature: float,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> Output:
        """Return logits and a sampled next token id."""
        output = self.forward(policy, seq)
        sampled = sample(output.logits, temperature, top_k, top_p, rng.generator())
        return Output(logits=output.logits, sampled_ids=sampled)

    def explain(self, seq: Seq, policy: Policy) -> dict[str, Any]:  # noqa: ARG002
        """Return interpretability data; trigram has none."""
        return {}
