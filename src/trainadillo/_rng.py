"""Random number generation: Generator class and module-level manual_seed."""

from typing import Self

import numpy as np


class _RngState:
    # Unset until trainadillo.manual_seed() is called. resolve_generator checks
    # for None and raises — unseeded random ops are a loud error, not silent
    # non-reproducibility.
    default: Generator | None = None


_state = _RngState()


class Generator:
    """Controllable RNG wrapping numpy's PCG64, with PyTorch's Generator interface.

    Generator() seeds from OS entropy — immediately usable, no manual_seed needed.
    Call manual_seed() to get a reproducible sequence. repr shows seeded/unseeded
    state for easy debugging.

    `np_rng` is the public numpy generator — callers use it to draw from the RNG.
    """

    def __init__(self) -> None:
        """Seed from OS entropy. Immediately usable; _seeded stays False."""
        self.np_rng: np.random.Generator = np.random.default_rng()
        self._seeded: bool = False

    def manual_seed(self, seed: int) -> Self:
        """Re-seed with an explicit integer seed. Returns self for chaining."""
        self.np_rng = np.random.Generator(np.random.PCG64(seed))
        self._seeded = True
        return self

    def __repr__(self) -> str:
        """State label — seeded/unseeded — not a constructor parameter."""
        state = "seeded" if self._seeded else "unseeded"
        return f"Generator({state})"


def resolve_generator(generator: Generator | None) -> Generator:
    """Return the generator to use, raising if none has been seeded."""
    if generator is not None:
        return generator
    if _state.default is None:
        msg = (
            "No random generator available. "
            "Call `trainadillo.manual_seed(seed)` before using random ops, "
            "or pass an explicit `generator=` argument."
        )
        raise RuntimeError(msg)
    return _state.default


def manual_seed(seed: int) -> Generator:
    """Seed the module-level default generator and return it.

    Matches torch.manual_seed(seed) — returns the Generator so callers can
    either use the default implicitly or keep a reference for explicit passing.
    """
    if _state.default is None:
        _state.default = Generator()
    _state.default.manual_seed(seed)
    return _state.default
