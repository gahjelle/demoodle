## Context

T1 is the foundation of trainadillo. Its only job is to establish a `Tensor` wrapper around `numpy.ndarray` — a layer where we can later attach metadata (gradients, graph nodes) that raw numpy arrays cannot hold. T1 builds purely the data layer; T8 adds the autograd layer on top.

This is a learning project: every design decision is explained in terms of *why PyTorch made this choice*, not just *what to build*.

## Goals / Non-Goals

**Goals:**
- `Tensor` wraps `np.ndarray` with a faithful subset of PyTorch's public interface
- `Size` matches `torch.Size` including repr format and `numel()`
- Arithmetic (forward + reflected), comparison, and indexing all work correctly
- No autograd fields — clean boundary; T8 extends the class

**Non-Goals:**
- Gradient tracking of any kind
- Creation functions (T2), random ops (T4)
- Any `__init__.py` / public API wiring (T18)

## Decisions

### `Size` is a `tuple` subclass with `numel()` and PyTorch repr

```python
class Size(tuple[int, ...]):
    def __repr__(self) -> str:
        return f"torch.Size([{', '.join(str(x) for x in self)}])"

    def numel(self) -> int:
        return math.prod(self)
```

`torch.Size` is PyTorch's return type for `.shape` and `.size()`. It behaves exactly like a tuple but has a distinctive repr and one extra method. Subclassing `tuple` gives all tuple semantics for free; `numel()` is a one-liner. The `torch.Size([...])` repr matches PyTorch output exactly, which is useful when the learning docs show side-by-side output comparisons.

### `__bool__` is deliberately absent

PyTorch raises `RuntimeError: Boolean value of Tensor with more than one element is ambiguous` when you write `if tensor:`. We match this by simply not implementing `__bool__`. Python then raises `TypeError` on any attempted bool conversion — not identical to PyTorch's message, but close enough.

**Why PyTorch forbids it:** without this guard, `if tensor:` on a weight matrix would silently evaluate the truthiness of the first element. Forcing `.item()` for scalars and `.any()`/`.all()` for arrays makes intent explicit. There is a second reason: boolean Tensors used as masks (e.g. `sorted_probs > top_p` → mask passed to `masked_fill`) must not accidentally collapse to Python bools during expression evaluation.

### Integer indexing always returns a 0-D Tensor, not a Python scalar

`tensor[3]` returns a `Tensor` with shape `()`, not a Python `float`. This preserves the invariant that indexing always produces a Tensor, which T11 depends on: if `weight[token_id]` returned a Python float, autograd would lose the gradient.

**Implementation note:** `np.array([1, 2, 3])[0]` returns a numpy scalar (`np.int64(1)`), not a 0-D array. We must wrap explicitly: `Tensor(np.asarray(self._data[key]))`.

### `view(*shape_or_dtype)` dispatches on argument type at runtime

```
tensor.view(3, 4)        # reshape  — args are ints
tensor.view(np.uint8)    # reinterpret bytes — arg is a numpy dtype
```

If the first argument is a numpy dtype (all trainadillo dtype constants are numpy dtype types), delegate to `ndarray.view(dtype)`. Otherwise delegate to `ndarray.reshape(*shape)`.

**Why this dual behaviour exists:** `persistence.py` uses `.view(torch.uint8)` to reinterpret float bytes as raw bytes for hashing. PyTorch uses the same method name for both operations; matching this quirk avoids changes to `persistence.py` during T20.

### Reflected arithmetic operators are included

`__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`, `__rmatmul__` alongside their forward counterparts. Numpy does the actual computation; we just delegate and wrap the result.

**Why:** Python calls `__radd__` on the right operand when the left operand's `__add__` returns `NotImplemented`. `2 * tensor` and `tensor * 2` must both work. Training code often writes `1.0 / tensor` or `scalar - tensor`.

### `__repr__` uses `numpy.array2string` with `precision=4`

```python
def __repr__(self) -> str:
    inner = np.array2string(self._data, precision=4, separator=', ')
    return f"tensor({inner})"
```

This matches PyTorch's `tensor([1.5000, 2.0000])` format for the common case. Edge cases (very small/large floats switching between fixed and scientific notation) may diverge slightly — accepted as "close enough" for a learning project. The repr only affects display; it has no bearing on correctness.

### Dtype constants are numpy dtype objects at module level

```python
long = np.int64
uint8 = np.uint8
float32 = np.float32
```

These are the actual numpy dtype classes, so `np.array(data, dtype=long)` works directly. This matches how PyTorch maps `torch.long → torch.int64` internally: PyTorch dtype objects are thin wrappers that ultimately resolve to C scalar types; our numpy dtypes play the same role.

### Precise numpy generics for type annotations

numpy 2.x (the project uses 2.4.6) supports proper generics. `_data` is typed as `np.ndarray[Any, np.dtype[np.generic]]` — the broadest correct form. Where the dtype is known (e.g. comparison ops always return `bool_`), use the specific form: `np.ndarray[Any, np.dtype[np.bool_]]`. Return types for methods that produce Tensors use `Tensor`. Constructor takes `np.ndarray[Any, np.dtype[np.generic]]`.

### Autograd fields are absent in T1

`grad`, `requires_grad`, `_grad_fn`, `_is_leaf` do not exist yet. T8 adds them when the computation graph machinery exists to enforce their invariants. Adding them in T1 would be dead weight with nothing to maintain the invariants.

## Risks / Trade-offs

- **Two-pass arithmetic dunders:** `__add__` etc. are non-differentiable in T1 and replaced in T10. Accepted: the plan documents this and the two-pass approach keeps T1 focused.
- **`view()` runtime dispatch:** detecting dtype vs shape args at runtime is slightly ad-hoc. Sufficient for the known callers in this codebase; a future caller with an unexpected argument type would raise a clear error.
- **`__bool__` raises `TypeError` not `RuntimeError`:** the error class differs from PyTorch. Acceptable; the intent (prevent silent bool conversion) is matched.

## Open Questions

None. All design decisions resolved during exploration.
