"""Shared sampling logic for all architecture implementations."""

import torch


def sample(
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
        scaled = torch.full_like(scaled, float("-inf")).scatter(
            0, top_indices, scaled[top_indices]
        )

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        to_remove = (cumulative - sorted_probs) > top_p
        sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
        scaled = torch.zeros_like(scaled).scatter(0, sorted_indices, sorted_logits)

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(0)
