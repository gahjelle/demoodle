"""Random tensor creation functions — rand and randint with explicit seeding."""

from typing import TYPE_CHECKING

import numpy as np

import trainadillo._rng as _rng_module
from trainadillo._tensor import Tensor

if TYPE_CHECKING:
    from trainadillo._rng import Generator


def _resolve_generator(generator: Generator | None) -> Generator:
    """Return the generator to use, raising if none has been seeded."""
    if generator is not None:
        return generator
    if _rng_module._default_generator is None:  # noqa: SLF001
        msg = (
            "No random generator available. "
            "Call `trainadillo.manual_seed(seed)` before using `rand` or `randint`, "
            "or pass an explicit `generator=` argument."
        )
        raise RuntimeError(msg)
    return _rng_module._default_generator  # noqa: SLF001


def rand(*shape: int, generator: Generator | None = None) -> Tensor:
    """Return a float32 Tensor of the given shape with values uniform in [0, 1).

    When generator is None, uses the module-level default generator — which must
    have been seeded via trainadillo.manual_seed() first, or a RuntimeError is raised.
    """
    rng = _resolve_generator(generator)
    return Tensor(rng._np_rng.random(shape).astype(np.float32))  # noqa: SLF001


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
    rng = _resolve_generator(generator)
    return Tensor(rng._np_rng.integers(low, high, size=size, dtype=np.int64))  # noqa: SLF001
