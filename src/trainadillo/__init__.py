"""Trainadillo: a minimal PyTorch clone built on NumPy, for learning."""

from trainadillo._autograd import GradFn, grad_enabled, no_grad
from trainadillo._random import rand, randint
from trainadillo._rng import Generator, manual_seed

__all__ = [
    "Generator",
    "GradFn",
    "grad_enabled",
    "manual_seed",
    "no_grad",
    "rand",
    "randint",
]
