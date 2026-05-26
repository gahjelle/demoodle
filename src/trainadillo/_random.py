"""Random tensor creation functions — rand and randint with explicit seeding."""

import numpy as np

from trainadillo._rng import Generator, resolve_generator
from trainadillo._tensor import Tensor


def rand(*shape: int, generator: Generator | None = None) -> Tensor:
    """Return a float32 Tensor of the given shape with values uniform in [0, 1).

    When generator is None, uses the module-level default generator — which must
    have been seeded via trainadillo.manual_seed() first, or a RuntimeError is raised.
    """
    rng = resolve_generator(generator)
    return Tensor(rng.np_rng.random(shape).astype(np.float32))


def randint(
    low: int,
    high: int,
    size: tuple[int, ...],
    *,
    generator: Generator | None = None,
) -> Tensor:
    """Return an int64 Tensor of shape `size` with values uniform in [low, high).

    size is a tuple — matches PyTorch's randint(low, high, size) signature.
    When generator is None, uses the module-level default generator (must be seeded).
    """
    rng = resolve_generator(generator)
    return Tensor(rng.np_rng.integers(low, high, size=size, dtype=np.int64))
