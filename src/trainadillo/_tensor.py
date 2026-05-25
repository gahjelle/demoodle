"""The Tensor type: numpy array wrapper with PyTorch-compatible API."""

from typing import Any

import numpy as np

from trainadillo._size import Size

type NDArray = np.ndarray[Any, np.dtype[np.generic]]

# Dtype constants — numpy scalar types, matching torch.long / torch.float32 etc.
# Lowercase to match PyTorch's naming convention (torch.long, not torch.LONG).
long: type[np.int64] = np.int64
uint8: type[np.uint8] = np.uint8
float32: type[np.float32] = np.float32


class Tensor:
    """Multidimensional array with a PyTorch-compatible interface.

    Wraps a numpy.ndarray and exposes shape metadata, arithmetic, comparison,
    and indexing operations. Autograd fields (grad, requires_grad, _grad_fn)
    are absent in T1 and added in T8 when the computation graph exists to use them.
    """

    __hash__: None = None  # Tensors are unhashable, matching PyTorch behaviour.

    def __init__(self, data: NDArray) -> None:
        """Wrap a numpy array. Scalars are promoted to 0-D arrays via np.asarray."""
        # Any here: numpy stubs don't support arithmetic on np.dtype[np.generic];
        # type is enforced at the public boundary (properties, method signatures).
        self._data: Any = np.asarray(data)

    # ------------------------------------------------------------------
    # Shape metadata
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Size:
        """Return the tensor's shape as a Size."""
        return Size(self._data.shape)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return int(self._data.ndim)

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """Return the element dtype."""
        return self._data.dtype

    @property
    def data(self) -> NDArray:
        """Return the underlying numpy array."""
        return self._data

    # ------------------------------------------------------------------
    # Extraction and conversion
    # ------------------------------------------------------------------

    def item(self) -> float | int | bool:
        """Extract a Python scalar from a 0-D or single-element tensor."""
        return self._data.item()

    def tolist(self) -> Any:  # noqa: ANN401 — inherently recursive; numpy types as Any
        """Convert to nested Python lists, or a Python scalar for 0-D tensors."""
        return self._data.tolist()

    def __len__(self) -> int:
        """Return the size of the first dimension."""
        return int(self._data.shape[0])

    def __repr__(self) -> str:
        """Return tensor(...) repr, matching PyTorch's format closely."""
        return f"tensor({np.array2string(self._data, precision=4, separator=', ')})"

    # ------------------------------------------------------------------
    # Size and shape manipulation
    # ------------------------------------------------------------------

    def size(self, dim: int | None = None) -> Size | int:
        """Return the full shape as a Size, or the size along one dimension."""
        s = Size(self._data.shape)
        return s if dim is None else s[dim]

    def cpu(self) -> Tensor:
        """No-op: trainadillo tensors are always on CPU."""
        return self

    def contiguous(self) -> Tensor:
        """No-op: numpy arrays are always in C-contiguous memory."""
        return self

    def view(self, *shape_or_dtype: int | type[np.generic]) -> Tensor:
        """Reshape or reinterpret raw bytes.

        Called with ints: reshape (like numpy.reshape).
        Called with a numpy dtype type: reinterpret bytes (like ndarray.view(dtype)).
        This dual behaviour mirrors torch.Tensor.view, needed for persistence hashing.
        """
        if (
            len(shape_or_dtype) == 1
            and isinstance(shape_or_dtype[0], type)
            and issubclass(shape_or_dtype[0], np.generic)
        ):
            return Tensor(self._data.view(shape_or_dtype[0]))
        shape = tuple(int(x) for x in shape_or_dtype)  # ty: ignore[invalid-argument-type]
        return Tensor(self._data.reshape(shape))

    def squeeze(self, dim: int | None = None) -> Tensor:
        """Remove dimensions of size 1."""
        if dim is None:
            return Tensor(self._data.squeeze())
        return Tensor(np.squeeze(self._data, axis=dim))

    def flatten(self) -> Tensor:
        """Return a 1-D copy."""
        return Tensor(self._data.flatten())

    # ------------------------------------------------------------------
    # Indexing — always returns a Tensor, never a Python scalar
    # ------------------------------------------------------------------

    def __getitem__(self, key: int | slice | Tensor) -> Tensor:
        """Index into the tensor.

        Always returns a Tensor. Integer indexing on a 1-D tensor produces a
        0-D Tensor (shape ()), not a Python scalar. This is critical for T11:
        weight[token_id] must remain in the gradient graph.
        """
        index = key._data if isinstance(key, Tensor) else key  # noqa: SLF001
        return Tensor(np.asarray(self._data[index]))

    # ------------------------------------------------------------------
    # Arithmetic operators — non-differentiable in T1; upgraded in T10
    # ------------------------------------------------------------------

    def __add__(self, other: Tensor | float) -> Tensor:
        """Element-wise addition."""
        return Tensor(self._data + _data(other))

    def __radd__(self, other: float) -> Tensor:
        """Right-hand addition: other + self."""
        return Tensor(other + self._data)

    def __sub__(self, other: Tensor | float) -> Tensor:
        """Element-wise subtraction."""
        return Tensor(self._data - _data(other))

    def __rsub__(self, other: float) -> Tensor:
        """Right-hand subtraction: other - self."""
        return Tensor(other - self._data)

    def __mul__(self, other: Tensor | float) -> Tensor:
        """Element-wise multiplication."""
        return Tensor(self._data * _data(other))

    def __rmul__(self, other: float) -> Tensor:
        """Right-hand multiplication: other * self."""
        return Tensor(other * self._data)

    def __truediv__(self, other: Tensor | float) -> Tensor:
        """Element-wise division."""
        return Tensor(self._data / _data(other))

    def __rtruediv__(self, other: float) -> Tensor:
        """Right-hand division: other / self."""
        return Tensor(other / self._data)

    def __neg__(self) -> Tensor:
        """Negate all elements."""
        return Tensor(-self._data)

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix multiplication."""
        return Tensor(self._data @ other._data)

    def __rmatmul__(self, other: Tensor) -> Tensor:
        """Right-hand matrix multiplication: other @ self."""
        return Tensor(other._data @ self._data)

    # ------------------------------------------------------------------
    # Comparison operators — return boolean Tensors, never Python bools.
    # __bool__ raises TypeError: converting a multi-element Tensor to bool
    # is ambiguous (matching PyTorch's RuntimeError behaviour).
    # ------------------------------------------------------------------

    def __gt__(self, other: Tensor | float) -> Tensor:
        """Element-wise greater-than; returns a boolean Tensor."""
        return Tensor(self._data > _data(other))

    def __ge__(self, other: Tensor | float) -> Tensor:
        """Element-wise greater-than-or-equal; returns a boolean Tensor."""
        return Tensor(self._data >= _data(other))

    def __lt__(self, other: Tensor | float) -> Tensor:
        """Element-wise less-than; returns a boolean Tensor."""
        return Tensor(self._data < _data(other))

    def __le__(self, other: Tensor | float) -> Tensor:
        """Element-wise less-than-or-equal; returns a boolean Tensor."""
        return Tensor(self._data <= _data(other))

    def __eq__(self, other: object) -> Tensor:  # ty: ignore[invalid-method-override]
        """Element-wise equality; returns a boolean Tensor."""
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data == other_data)

    def __ne__(self, other: object) -> Tensor:  # ty: ignore[invalid-method-override]
        """Element-wise inequality; returns a boolean Tensor."""
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data != other_data)

    def __bool__(self) -> bool:
        """Raise TypeError — boolean conversion of a Tensor is always ambiguous.

        PyTorch raises RuntimeError for the same reason: which element's truthiness
        would a multi-element Tensor represent? Forcing .item() or .any()/.all()
        makes intent explicit. We raise TypeError (instead of RuntimeError) because
        Python's bool() protocol expects TypeError for unsupported conversions.
        """
        msg = "Boolean value of Tensor is ambiguous. Use .item(), .any(), or .all()."
        raise TypeError(msg)


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------


def _data(value: Tensor | float) -> NDArray | float:
    """Unwrap a Tensor to its underlying numpy array, or pass scalars through."""
    return value._data if isinstance(value, Tensor) else value  # noqa: SLF001
