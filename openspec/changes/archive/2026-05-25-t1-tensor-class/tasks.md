## 1. Create `Size` class

- [x] 1.1 Create `src/trainadillo/_size.py` with `Size(tuple)`: `__repr__` → `torch.Size([...])`, `numel()` → `math.prod(self)`; annotate `__new__` to accept an iterable of ints

## 2. Create `Tensor` class and dtype constants

- [x] 2.1 Create `src/trainadillo/_tensor.py`; add module-level dtype constants `long = np.int64`, `uint8 = np.uint8`, `float32 = np.float32`
- [x] 2.2 `Tensor.__init__`: accept `np.ndarray[Any, np.dtype[np.generic]]`; store as `_data`
- [x] 2.3 Properties: `shape -> Size`, `ndim -> int`, `dtype -> np.dtype[np.generic]`, `data -> np.ndarray[Any, np.dtype[np.generic]]`
- [x] 2.4 `item() -> float | int | bool` — extract Python scalar; `tolist()` — delegate to numpy; `__len__() -> int` — return `self._data.shape[0]`
- [x] 2.5 `size(dim: int | None = None) -> Size | int` — return `shape` or `shape[dim]`
- [x] 2.6 `cpu() -> Tensor` and `contiguous() -> Tensor` — no-ops returning `self`
- [x] 2.7 `view(*shape_or_dtype)` — if first arg is a numpy dtype, delegate to `_data.view(dtype)`; otherwise delegate to `_data.reshape(*shape)`
- [x] 2.8 `squeeze(dim: int | None = None) -> Tensor`, `flatten() -> Tensor` — delegate to numpy
- [x] 2.9 `__getitem__` — return a `Tensor`; ensure integer index on 1-D array wraps result in `np.asarray()` to produce a 0-D array, not a numpy scalar
- [x] 2.10 Forward arithmetic dunders — `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__neg__`, `__matmul__` — delegate to numpy, wrap result in `Tensor`
- [x] 2.11 Reflected arithmetic dunders — `__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`, `__rmatmul__` — same pattern, operands reversed
- [x] 2.12 Comparison dunders — `__gt__`, `__ge__`, `__lt__`, `__le__`, `__eq__` — delegate to numpy, wrap boolean result in `Tensor`; **do not** implement `__bool__`
- [x] 2.13 `__repr__` — `f"tensor({np.array2string(self._data, precision=4, separator=', ')})"` 

## 3. Tests

- [x] 3.1 `shape`, `ndim`, `dtype`, `data` return correct values for 1-D and 2-D arrays
- [x] 3.2 `item()` returns a Python `float`/`int`; `tolist()` returns a Python list
- [x] 3.3 `size()` returns a `Size` matching `shape`; `size(dim)` returns an `int`; `Size.numel()` returns the product of dims; `repr(Size([3, 4])) == "torch.Size([3, 4])"`
- [x] 3.4 `view(3, 4)` reshapes a 12-element tensor; `view(np.uint8)` on a float32 tensor returns a byte tensor with 4× the element count
- [x] 3.5 `squeeze` removes a length-1 dimension; `flatten` returns a 1-D tensor
- [x] 3.6 `__getitem__` with a slice returns a Tensor; integer index on a 1-D tensor returns a 0-D Tensor (not a numpy scalar); `t[0].shape == ()`
- [x] 3.7 `tensor + tensor`, `tensor - tensor`, `tensor * 3`, `3 * tensor` (radd), `tensor / 2.0`, `-tensor`, 2-D `@` matmul all return Tensors with correct values
- [x] 3.8 `tensor + int` works (int on right); `int + tensor` works (int on left via `__radd__`)
- [x] 3.9 `tensor > scalar` returns a boolean Tensor; that boolean Tensor can be used to index another Tensor
- [x] 3.10 `bool(tensor)` raises `TypeError` for a multi-element Tensor
- [x] 3.11 `Tensor` constructed with `dtype=long` has `dtype == np.int64`; with `dtype=float32` has `dtype == np.float32`
- [x] 3.12 `repr(tensor)` starts with `"tensor("` and ends with `")"` 
- [x] 3.13 `len(tensor)` returns the length of the first dimension

## 4. Learning doc

- [x] 4.1 Write `docs/trainadillo/T1-tensor-class.md` covering: the concept (why wrap numpy at all), `Size` and its relation to `torch.Size`, why `__bool__` is absent, why integer indexing returns a 0-D Tensor, how `view()` dual-dispatch works, what dtype constants are and how PyTorch uses them internally, and what T8 will add on top of this foundation

## 5. Verification

- [x] 5.1 `uv run ruff format src/trainadillo/ tests/trainadillo/`
- [x] 5.2 `uv run ruff check src/trainadillo/ tests/trainadillo/`
- [x] 5.3 `uv run ty check src/trainadillo/ tests/trainadillo/`
- [x] 5.4 `uv run pytest tests/trainadillo/`

## 6. Documentation

- [x] 6.1 Mark T1 done (✅) in `PLANS_TRAINADILLO.md`
