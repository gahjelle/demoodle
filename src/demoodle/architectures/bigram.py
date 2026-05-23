"""Learned-bigram architecture: a V x V weight matrix as the entire model."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from demoodle.core.types import Output, Policy, Seq

if TYPE_CHECKING:
    from demoodle.core.rng import RNG


class BigramModel(nn.Module):
    """V x V parameter: row N is the next-token distribution after seeing token N."""

    def __init__(self, vocab_size: int) -> None:
        """Initialise with a zero-filled (vocab_size, vocab_size) weight matrix."""
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(vocab_size, vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the row of the weight matrix corresponding to token x."""
        return self.weight[x]


def _sample(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token from logits with temperature, top-k, and nucleus filtering."""
    scaled = logits / temperature

    if top_k is not None:
        k = min(top_k, scaled.size(-1))
        _, top_indices = torch.topk(scaled, k)
        mask = torch.full_like(scaled, float("-inf"))
        mask.scatter_(0, top_indices, scaled[top_indices])
        scaled = mask

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        to_remove = (cumulative - sorted_probs) > top_p
        sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
        scaled = torch.zeros_like(scaled).scatter_(0, sorted_indices, sorted_logits)

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(0)


@dataclass(frozen=True)
class BigramArchitecture:
    """V x V weight-matrix model. Stateless: no Policy held internally."""

    vocab_size: int
    context_length: int = 1

    def init_state(self, rng: RNG) -> Policy:
        """Return a freshly initialised Policy. Pure function of rng."""
        model = BigramModel(self.vocab_size)
        nn.init.normal_(model.weight, generator=rng.generator())
        return Policy(model=model)

    def forward(self, policy: Policy, tokens: Seq) -> Output:
        """Run a forward pass using only the last token."""
        logits: torch.Tensor = policy.model(tokens[-1])
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
        sampled = _sample(output.logits, temperature, top_k, top_p, rng.generator())
        return Output(logits=output.logits, sampled_ids=sampled)

    def explain(self, seq: Seq, policy: Policy) -> dict[str, Any]:  # noqa: ARG002
        """Return interpretability data; bigram has none."""
        return {}
