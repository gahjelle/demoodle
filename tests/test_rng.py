import dataclasses

import pytest
import torch

from demoodle.core.rng import RNG

# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_rng_holds_seed() -> None:
    rng = RNG(seed=42)
    assert rng.seed == 42


def test_rng_is_frozen() -> None:
    rng = RNG(seed=42)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rng.seed = 99  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def test_split_is_deterministic() -> None:
    rng = RNG(seed=42)
    left_a, right_a = rng.split()
    left_b, right_b = rng.split()
    assert left_a.seed == left_b.seed
    assert right_a.seed == right_b.seed


def test_split_children_diverge() -> None:
    rng = RNG(seed=42)
    left, right = rng.split()
    assert left.seed != right.seed


def test_split_different_parents_produce_different_children() -> None:
    left_a, right_a = RNG(seed=1).split()
    left_b, right_b = RNG(seed=2).split()
    assert left_a.seed != left_b.seed
    assert right_a.seed != right_b.seed


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------


def test_generator_is_reproducible() -> None:
    rng = RNG(seed=42)
    draws_a = torch.rand(5, generator=rng.generator())
    draws_b = torch.rand(5, generator=rng.generator())
    assert torch.equal(draws_a, draws_b)


def test_different_seeds_produce_different_draws() -> None:
    draws_a = torch.rand(5, generator=RNG(seed=1).generator())
    draws_b = torch.rand(5, generator=RNG(seed=2).generator())
    assert not torch.equal(draws_a, draws_b)
