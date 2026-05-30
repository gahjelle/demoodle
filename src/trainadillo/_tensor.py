"""The Tensor type: numpy array wrapper with PyTorch-compatible API."""

from typing import TYPE_CHECKING, Any, Self

import numpy as np

from trainadillo._size import Size

if TYPE_CHECKING:
    from trainadillo._autograd import GradFn

type NDArray = np.ndarray[Any, np.dtype[np.generic]]

# Dtype constants — numpy scalar types, matching torch.long / torch.float32 etc.
# Lowercase to match PyTorch's naming convention (torch.long, not torch.LONG).
long: type[np.int64] = np.int64
uint8: type[np.uint8] = np.uint8
float32: type[np.float32] = np.float32


class Tensor:
    """Multidimensional array with a PyTorch-compatible interface.

    Wraps a numpy.ndarray and exposes shape metadata, arithmetic, comparison,
    and indexing operations. Autograd fields (grad, requires_grad, grad_fn,
    is_leaf) are absent in T1 and added in T8 when the computation graph exists.
    """

    __hash__: None = None  # Tensors are unhashable, matching PyTorch behaviour.

    def __init__(self, data: NDArray, *, requires_grad: bool = False) -> None:
        """Wrap a numpy array. Scalars are promoted to 0-D arrays via np.asarray."""
        # Any here: numpy stubs don't support arithmetic on np.dtype[np.generic];
        # type is enforced at the public boundary (properties, method signatures).
        self.data: Any = np.asarray(data)
        self.requires_grad: bool = requires_grad
        self.grad: Tensor | None = None
        self.grad_fn: GradFn | None = None

    # ------------------------------------------------------------------
    # Autograd
    # ------------------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        """True when this tensor was created directly, not by an operation.

        A tensor is a leaf iff grad_fn is None. Parameters are always leaves;
        intermediate results of differentiable ops are not.
        """
        return self.grad_fn is None

    def detach(self) -> Tensor:
        """Return a tensor sharing data but outside the computation graph."""
        return Tensor(self.data)

    def backward(self) -> None:
        """Walk the computation graph and accumulate gradients into leaves.

        The backward pass has three phases:
          1. DFS post-order from self.grad_fn to collect all GradFn nodes.
          2. Seed: assign gradient 1.0 to this (scalar) tensor's grad_fn.
          3. Walk in reverse topological order, calling each GradFn.backward()
             and routing contributions to leaves (.grad) or upstream buffers.
        """
        if self.data.shape != ():
            msg = "backward() can only be called on scalar tensors"
            raise RuntimeError(msg)

        if self.grad_fn is None:
            if self.requires_grad:
                self.grad = Tensor(np.ones_like(self.data))
            return

        topo: list[GradFn] = []
        _topo_dfs(self.grad_fn, set(), topo)

        accumulated: dict[int, Any] = {id(self.grad_fn): np.ones(())}
        for node in reversed(topo):
            grad = accumulated[id(node)]
            for tensor, grad_contrib in node.backward(grad):
                _accumulate(tensor, grad_contrib, accumulated)

    # ------------------------------------------------------------------
    # Shape metadata
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Size:
        """Return the tensor's shape as a Size."""
        return Size(self.data.shape)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return int(self.data.ndim)

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """Return the element dtype."""
        return self.data.dtype

    # ------------------------------------------------------------------
    # Extraction and conversion
    # ------------------------------------------------------------------

    def item(self) -> float | int | bool:
        """Extract a Python scalar from a 0-D or single-element tensor."""
        return self.data.item()

    def tolist(self) -> Any:  # noqa: ANN401
        """Convert to nested Python lists, or a Python scalar for 0-D tensors."""
        return self.data.tolist()

    def __len__(self) -> int:
        """Return the size of the first dimension."""
        return int(self.data.shape[0])

    def __repr__(self) -> str:
        """Return tensor(...) repr, matching PyTorch's format closely."""
        return f"tensor({np.array2string(self.data, precision=4, separator=', ')})"

    # ------------------------------------------------------------------
    # Size and shape manipulation
    # ------------------------------------------------------------------

    def size(self, dim: int | None = None) -> Size | int:
        """Return the full shape as a Size, or the size along one dimension."""
        s = Size(self.data.shape)
        return s if dim is None else s[dim]

    def cpu(self) -> Self:
        """No-op: trainadillo tensors are always on CPU."""
        return self

    def contiguous(self) -> Self:
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
            return Tensor(self.data.view(shape_or_dtype[0]))
        shape = tuple(int(x) for x in shape_or_dtype)  # ty: ignore[invalid-argument-type]
        return Tensor(self.data.reshape(shape))

    def squeeze(self, dim: int | None = None) -> Tensor:
        """Remove dimensions of size 1."""
        if dim is None:
            return Tensor(self.data.squeeze())
        return Tensor(np.squeeze(self.data, axis=dim))

    def flatten(self) -> Tensor:
        """Return a 1-D copy."""
        return Tensor(self.data.flatten())

    def masked_fill(self, mask: Tensor, value: float) -> Tensor:
        """Return a new Tensor with True positions in mask replaced by value.

        self is not modified. Used in nucleus sampling to zero out low-probability
        tokens by filling their logits with float("-inf") before softmax.
        """
        out = self.data.copy()
        out[mask.data] = value
        return Tensor(out)

    def scatter(self, dim: int, index: Tensor, src: Tensor) -> Tensor:
        """Return a new Tensor with src values written at index positions.

        For dim=0 on 1-D tensors: result[index[i]] = src[i]. self is not modified.
        Only dim=0 is implemented; the N-D general case is deferred to T28.
        """
        if dim != 0:
            msg = f"scatter: dim={dim} not implemented; only dim=0 supported"
            raise NotImplementedError(msg)
        out = self.data.copy()
        out[index.data] = src.data
        return Tensor(out)

    def argmax(self, dim: int | None = None) -> Tensor:
        """Return the index of the maximum value.

        When dim is None, operates over the flattened tensor and returns a 0-D
        Tensor. When dim is given, returns a Tensor with that dimension removed.
        Matches torch.Tensor.argmax behaviour.
        """
        return Tensor(np.argmax(self.data, axis=dim))

    # ------------------------------------------------------------------
    # Indexing — always returns a Tensor, never a Python scalar
    # ------------------------------------------------------------------

    def __getitem__(self, key: int | slice | Tensor) -> Tensor:
        """Index into the tensor.

        Always returns a Tensor. Integer indexing on a 1-D tensor produces a
        0-D Tensor (shape ()), not a Python scalar. This is critical for T11:
        weight[token_id] must remain in the gradient graph.
        """
        index = key.data if isinstance(key, Tensor) else key
        return Tensor(np.asarray(self.data[index]))

    # ------------------------------------------------------------------
    # Arithmetic operators — non-differentiable in T1; upgraded in T10
    # ------------------------------------------------------------------

    def __add__(self, other: Tensor | float) -> Tensor:
        """Element-wise addition."""
        return Tensor(self.data + _unwrap(other))

    def __radd__(self, other: float) -> Tensor:
        """Right-hand addition: other + self."""
        return Tensor(other + self.data)

    def __sub__(self, other: Tensor | float) -> Tensor:
        """Element-wise subtraction."""
        return Tensor(self.data - _unwrap(other))

    def __rsub__(self, other: float) -> Tensor:
        """Right-hand subtraction: other - self."""
        return Tensor(other - self.data)

    def __mul__(self, other: Tensor | float) -> Tensor:
        """Element-wise multiplication."""
        return Tensor(self.data * _unwrap(other))

    def __rmul__(self, other: float) -> Tensor:
        """Right-hand multiplication: other * self."""
        return Tensor(other * self.data)

    def __truediv__(self, other: Tensor | float) -> Tensor:
        """Element-wise division."""
        return Tensor(self.data / _unwrap(other))

    def __rtruediv__(self, other: float) -> Tensor:
        """Right-hand division: other / self."""
        return Tensor(other / self.data)

    def __neg__(self) -> Tensor:
        """Negate all elements."""
        return Tensor(-self.data)

    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix multiplication."""
        return Tensor(self.data @ other.data)

    def __rmatmul__(self, other: Tensor) -> Tensor:
        """Right-hand matrix multiplication: other @ self."""
        return Tensor(other.data @ self.data)

    # ------------------------------------------------------------------
    # Comparison operators — return boolean Tensors, never Python bools.
    # __bool__ raises TypeError: converting a multi-element Tensor to bool
    # is ambiguous (matching PyTorch's RuntimeError behaviour).
    # ------------------------------------------------------------------

    def __gt__(self, other: Tensor | float) -> Tensor:
        """Element-wise greater-than; returns a boolean Tensor."""
        return Tensor(self.data > _unwrap(other))

    def __ge__(self, other: Tensor | float) -> Tensor:
        """Element-wise greater-than-or-equal; returns a boolean Tensor."""
        return Tensor(self.data >= _unwrap(other))

    def __lt__(self, other: Tensor | float) -> Tensor:
        """Element-wise less-than; returns a boolean Tensor."""
        return Tensor(self.data < _unwrap(other))

    def __le__(self, other: Tensor | float) -> Tensor:
        """Element-wise less-than-or-equal; returns a boolean Tensor."""
        return Tensor(self.data <= _unwrap(other))

    def __eq__(self, other: object) -> Tensor:  # ty: ignore[invalid-method-override]
        """Element-wise equality; returns a boolean Tensor."""
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data == other_data)

    def __ne__(self, other: object) -> Tensor:  # ty: ignore[invalid-method-override]
        """Element-wise inequality; returns a boolean Tensor."""
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data != other_data)

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
# Internal helpers
# ------------------------------------------------------------------


def _topo_dfs(node: GradFn, visited: set[int], topo: list[GradFn]) -> None:
    """Recursive DFS post-order traversal; appends nodes to topo."""
    if id(node) in visited:
        return
    visited.add(id(node))
    for inp in node.inputs:
        if inp.grad_fn is not None:
            _topo_dfs(inp.grad_fn, visited, topo)
    topo.append(node)


def _accumulate(tensor: Tensor, contrib: NDArray, accumulated: dict[int, Any]) -> None:
    """Route a gradient contribution to a leaf's .grad or an upstream buffer."""
    if tensor.is_leaf:
        if tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = Tensor(np.zeros_like(tensor.data))
            tensor.grad = Tensor(tensor.grad.data + contrib)
    elif tensor.grad_fn is not None:
        fn_id = id(tensor.grad_fn)
        if fn_id not in accumulated:
            accumulated[fn_id] = np.zeros_like(contrib)
        accumulated[fn_id] = accumulated[fn_id] + contrib


def _unwrap(value: Tensor | float) -> NDArray | float:
    """Unwrap a Tensor to its underlying numpy array, or pass scalars through."""
    return value.data if isinstance(value, Tensor) else value
