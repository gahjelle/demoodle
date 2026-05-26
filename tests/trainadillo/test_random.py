"""Tests for trainadillo._random: rand and randint."""

import numpy as np
import pytest

import trainadillo
import trainadillo._rng as _rng_module
from trainadillo import rand, randint
from trainadillo._rng import Generator
from trainadillo._tensor import Tensor


@pytest.fixture(autouse=True)
def reset_default_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the module-level default generator to None before each test."""
    monkeypatch.setattr(_rng_module, "_default_generator", None)


# --- rand ---


def test_rand_shape_and_dtype() -> None:
    g = Generator().manual_seed(0)
    result = rand(3, 4, generator=g)
    assert isinstance(result, Tensor)
    assert result.shape == (3, 4)
    assert result.dtype == np.float32


def test_rand_values_in_range() -> None:
    g = Generator().manual_seed(1)
    result = rand(1000, generator=g)
    assert float(result._data.min()) >= 0.0
    assert float(result._data.max()) < 1.0


def test_rand_same_seed_reproducible() -> None:
    g1 = Generator().manual_seed(42)
    g2 = Generator().manual_seed(42)
    assert np.array_equal(rand(5, generator=g1)._data, rand(5, generator=g2)._data)


def test_rand_different_seeds_diverge() -> None:
    g1 = Generator().manual_seed(1)
    g2 = Generator().manual_seed(2)
    assert not np.array_equal(rand(5, generator=g1)._data, rand(5, generator=g2)._data)


# --- randint ---


def test_randint_shape_and_dtype() -> None:
    g = Generator().manual_seed(0)
    result = randint(0, 10, (5,), generator=g)
    assert isinstance(result, Tensor)
    assert result.shape == (5,)
    assert result.dtype == np.int64


def test_randint_values_in_range() -> None:
    g = Generator().manual_seed(2)
    result = randint(0, 10, (1000,), generator=g)
    assert int(result._data.min()) >= 0
    assert int(result._data.max()) < 10


def test_randint_same_seed_reproducible() -> None:
    g1 = Generator().manual_seed(99)
    g2 = Generator().manual_seed(99)
    assert np.array_equal(
        randint(0, 100, (10,), generator=g1)._data,
        randint(0, 100, (10,), generator=g2)._data,
    )


def test_randint_different_seeds_diverge() -> None:
    g1 = Generator().manual_seed(3)
    g2 = Generator().manual_seed(4)
    assert not np.array_equal(
        randint(0, 100, (10,), generator=g1)._data,
        randint(0, 100, (10,), generator=g2)._data,
    )


# --- unseeded default raises ---


def test_rand_raises_when_no_default_generator() -> None:
    with pytest.raises(RuntimeError, match="manual_seed"):
        rand(3)


def test_randint_raises_when_no_default_generator() -> None:
    with pytest.raises(RuntimeError, match="manual_seed"):
        randint(0, 10, (3,))


def test_rand_succeeds_after_manual_seed() -> None:
    trainadillo.manual_seed(7)
    result = rand(3)
    assert result.shape == (3,)


def test_randint_succeeds_after_manual_seed() -> None:
    trainadillo.manual_seed(7)
    result = randint(0, 5, (3,))
    assert result.shape == (3,)


# --- explicit generator bypasses unseeded default ---


def test_rand_explicit_generator_bypasses_unseeded_default() -> None:
    g = Generator().manual_seed(5)
    result = rand(3, generator=g)
    assert result.shape == (3,)


def test_randint_explicit_generator_bypasses_unseeded_default() -> None:
    g = Generator().manual_seed(5)
    result = randint(0, 10, (3,), generator=g)
    assert result.shape == (3,)
