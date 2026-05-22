"""Explicit, immutable RNG for deterministic randomness threading."""

import hashlib
from dataclasses import dataclass

import torch


def _mix(seed: int, tag: int) -> int:
    digest = hashlib.sha256(f"{seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass(frozen=True)
class RNG:
    """Deterministic RNG value. Thread through the pipeline; never mutate."""

    seed: int

    def split(self) -> tuple[RNG, RNG]:
        """Return two independent child RNGs derived from this seed."""
        return RNG(seed=_mix(self.seed, 0)), RNG(seed=_mix(self.seed, 1))

    def generator(self) -> torch.Generator:
        """Return a seeded torch.Generator for use in random torch ops."""
        g = torch.Generator()
        g.manual_seed(self.seed & 0xFFFF_FFFF_FFFF_FFFF)
        return g
