"""The Size type: an immutable integer tuple matching torch.Size."""

import math
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from collections.abc import Iterable


class Size(tuple[int, ...]):
    """Immutable sequence of tensor dimension sizes, matching torch.Size."""

    __slots__ = ()

    def __new__(cls, iterable: Iterable[int] = ()) -> Self:
        """Construct from an iterable of ints."""
        return super().__new__(cls, iterable)

    def __repr__(self) -> str:
        """Return torch.Size([...]) repr, matching PyTorch output exactly."""
        return f"torch.Size([{', '.join(str(x) for x in self)}])"

    def numel(self) -> int:
        """Return the product of all dimensions."""
        return math.prod(self)
