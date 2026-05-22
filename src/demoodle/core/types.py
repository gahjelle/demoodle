"""Immutable value types that flow between pipeline stages."""

from dataclasses import dataclass, field

import torch

type Seq = torch.Tensor  # A 1D integer tensor of token IDs


@dataclass(frozen=True)
class Output:
    """Result of a model forward pass."""

    logits: torch.Tensor
    sampled_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class Corpus:
    """Raw text corpus — unsegmented, as loaded from disk."""

    text: str


@dataclass(frozen=True)
class Dataset:
    """Encoded corpus as a flat sequence of token IDs."""

    tokens: torch.Tensor


@dataclass(frozen=True)
class Policy:
    """Trained model. Write-once: never mutate after creation."""

    model: torch.nn.Module
    # Reserved for PPO (W25); None for all other training regimes
    value_head: torch.nn.Module | None = field(default=None)


@dataclass(frozen=True)
class Metrics:
    """Training history — per-step loss values."""

    losses: list[float]


# Artifact is the tagged union of all pipeline values.
# CharTokenizer (W8) and BpeTokenizer (W17) join when implemented.
# RewardModel and PreferenceData will be added in Milestone 5 (W21).
type Artifact = Corpus | Dataset | Policy | Metrics
