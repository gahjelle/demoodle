"""Factory functions for constructing Tensor objects — the torch.* creation API."""

from typing import TYPE_CHECKING, Any

import numpy as np

from trainadillo._tensor import Tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

type NDArray = np.ndarray[Any, np.dtype[np.generic]]


def tensor(data: object, *, dtype: type[np.generic] | None = None) -> Tensor:
    """Construct a Tensor by copying the input data.

    Accepts Python lists, scalars, NumPy arrays, or existing Tensors.
    When dtype is None and the inferred dtype is float64, downgrades to float32
    to match PyTorch's default dtype behaviour.
    """
    raw: NDArray = data._data if isinstance(data, Tensor) else np.array(data)  # noqa: SLF001
    if dtype is not None:
        return Tensor(raw.astype(dtype, copy=True))
    # Downgrade float64 → float32: numpy infers float64 for Python floats, but
    # PyTorch defaults to float32. Training code written against PyTorch expects
    # float32 activations and parameters.
    if raw.dtype == np.float64:
        return Tensor(raw.astype(np.float32, copy=True))
    return Tensor(raw.copy())


def zeros(*shape: int) -> Tensor:
    """Return a float32 Tensor of the given shape filled with zeros."""
    return Tensor(np.zeros(shape, dtype=np.float32))


def ones(*shape: int) -> Tensor:
    """Return a float32 Tensor of the given shape filled with ones."""
    return Tensor(np.ones(shape, dtype=np.float32))


def zeros_like(t: Tensor) -> Tensor:
    """Return a Tensor of the same shape and dtype as t, filled with zeros."""
    return Tensor(np.zeros_like(t._data))  # noqa: SLF001


def full_like(t: Tensor, value: float) -> Tensor:
    """Return a Tensor of the same shape and dtype as t, filled with value."""
    return Tensor(np.full_like(t._data, value))  # noqa: SLF001


def arange(n: int) -> Tensor:
    """Return a 1-D int64 Tensor containing [0, 1, ..., n-1].

    Explicit dtype=np.int64 avoids relying on platform-native integer width,
    which is int32 on Windows 64-bit.
    """
    return Tensor(np.arange(n, dtype=np.int64))


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    """Stack a sequence of Tensors along a new axis at dim."""
    arrays = [t._data for t in tensors]  # noqa: SLF001
    return Tensor(np.stack(arrays, axis=dim))


def equal(a: Tensor, b: Tensor) -> bool:
    """Return True iff a and b have the same shape and all elements are equal.

    This is a full reduction to a Python bool — torch.equal semantics.
    It is NOT the same as a == b, which returns an element-wise boolean Tensor.
    """
    return bool(np.array_equal(a._data, b._data))  # noqa: SLF001
