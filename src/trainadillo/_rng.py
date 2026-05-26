"""Random number generation: Generator class and module-level manual_seed."""

from typing import Self

import numpy as np

# Unset until trainadillo.manual_seed() is called. T4's rand/randint check for
# None and raise — unseeded random ops are a loud error, not silent non-reproducibility.
_default_generator: Generator | None = None


class Generator:
    """Controllable RNG wrapping numpy's PCG64, with PyTorch's Generator interface.

    Generator() seeds from OS entropy — immediately usable, no manual_seed needed.
    Call manual_seed() to get a reproducible sequence. repr shows seeded/unseeded
    state for easy debugging.

    T4's rand() and randint() access _np_rng directly; it is semi-private plumbing.
    """

    def __init__(self) -> None:
        """Seed from OS entropy. Immediately usable; _seeded stays False."""
        self._np_rng: np.random.Generator = np.random.default_rng()
        self._seeded: bool = False

    def manual_seed(self, seed: int) -> Self:
        """Re-seed with an explicit integer seed. Returns self for chaining."""
        self._np_rng = np.random.Generator(np.random.PCG64(seed))
        self._seeded = True
        return self

    def __repr__(self) -> str:
        """State label — seeded/unseeded — not a constructor parameter."""
        state = "seeded" if self._seeded else "unseeded"
        return f"Generator({state})"


def manual_seed(seed: int) -> Generator:
    """Seed the module-level default generator and return it.

    Matches torch.manual_seed(seed) — returns the Generator so callers can
    either use the default implicitly or keep a reference for explicit passing.
    """
    global _default_generator  # noqa: PLW0603
    if _default_generator is None:
        _default_generator = Generator()
    _default_generator.manual_seed(seed)
    return _default_generator
