"""Trainadillo: a minimal PyTorch clone built on NumPy, for learning."""

from trainadillo._random import rand, randint
from trainadillo._rng import Generator, manual_seed

__all__ = ["Generator", "manual_seed", "rand", "randint"]
