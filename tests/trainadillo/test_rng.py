"""Tests for trainadillo._rng (Generator class and module-level manual_seed)."""

import importlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator as IteratorGen

import trainadillo._rng as rng_module
from trainadillo._rng import Generator


@pytest.fixture(autouse=True)
def isolate_default_generator() -> IteratorGen:
    """Reset _default_generator to None before and after every test.

    Prevents module-level state from leaking between tests regardless of
    execution order. Tests that need the default set call manual_seed() themselves.
    """
    rng_module._state.default = None
    yield
    rng_module._state.default = None


# ---------------------------------------------------------------------------
# 2.2 — Generator constructs without error
# ---------------------------------------------------------------------------


def test_generator_constructs_without_error() -> None:
    g = Generator()
    assert isinstance(g.np_rng, np.random.Generator)


# ---------------------------------------------------------------------------
# 2.3 — Same seed reproduces same sequence
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_sequence() -> None:
    g = Generator()
    g.manual_seed(42)
    draws1 = g.np_rng.random(5).tolist()
    g.manual_seed(42)
    draws2 = g.np_rng.random(5).tolist()
    assert draws1 == draws2


# ---------------------------------------------------------------------------
# 2.4 — Different seeds diverge
# ---------------------------------------------------------------------------


def test_different_seeds_diverge() -> None:
    g1 = Generator()
    g1.manual_seed(42)
    g2 = Generator()
    g2.manual_seed(99)
    assert g1.np_rng.random(5).tolist() != g2.np_rng.random(5).tolist()


# ---------------------------------------------------------------------------
# 2.5 — manual_seed returns self
# ---------------------------------------------------------------------------


def test_manual_seed_returns_self() -> None:
    g = Generator()
    assert g.manual_seed(42) is g


# ---------------------------------------------------------------------------
# 2.6 — repr before seeding
# ---------------------------------------------------------------------------


def test_repr_unseeded() -> None:
    assert repr(Generator()) == "Generator(unseeded)"


# ---------------------------------------------------------------------------
# 2.7 — repr after seeding
# ---------------------------------------------------------------------------


def test_repr_seeded() -> None:
    g = Generator()
    g.manual_seed(42)
    assert repr(g) == "Generator(seeded)"


# ---------------------------------------------------------------------------
# 2.8 — module-level manual_seed returns a seeded Generator
# ---------------------------------------------------------------------------


def test_module_manual_seed_returns_seeded_generator() -> None:
    g = rng_module.manual_seed(42)
    assert repr(g) == "Generator(seeded)"


# ---------------------------------------------------------------------------
# 2.9 — module-level manual_seed returns the default generator
# ---------------------------------------------------------------------------


def test_module_manual_seed_returns_default_generator() -> None:
    g = rng_module.manual_seed(42)
    assert g is rng_module._state.default


# ---------------------------------------------------------------------------
# 2.9 — same seed twice reproduces sequence via module default
# ---------------------------------------------------------------------------


def test_module_manual_seed_same_seed_reproduces_sequence() -> None:
    rng_module.manual_seed(42)
    assert rng_module._state.default is not None
    draws1 = rng_module._state.default.np_rng.random(5).tolist()
    rng_module.manual_seed(42)
    assert rng_module._state.default is not None
    draws2 = rng_module._state.default.np_rng.random(5).tolist()
    assert draws1 == draws2


# ---------------------------------------------------------------------------
# 2.10 — default is None in a freshly imported module
# ---------------------------------------------------------------------------


def test_default_generator_starts_as_none() -> None:
    # Reload verifies the module-level initialisation, not just the fixture.
    importlib.reload(rng_module)
    assert rng_module._state.default is None


# ---------------------------------------------------------------------------
# 2.11 — default is set after manual_seed
# ---------------------------------------------------------------------------


def test_default_generator_set_after_manual_seed() -> None:
    rng_module.manual_seed(42)
    assert rng_module._state.default is not None
