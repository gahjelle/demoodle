# T2 — Creation Functions

## The concept: factory functions as the user-facing API

In T1 we built `Tensor` — a raw wrapper around a numpy array. But constructing one
requires importing numpy directly:

```python
import numpy as np
from trainadillo._tensor import Tensor

t = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
```

That is too much ceremony. Real PyTorch code looks like this:

```python
import torch

t = torch.tensor([1.0, 2.0, 3.0])
```

T2 adds those factory functions. They are thin wrappers around numpy calls, but they
make three important guarantees that bare numpy doesn't:

1. **Predictable dtypes** — no platform-specific surprises
2. **Always-copy semantics** for `tensor()` — no aliasing footguns
3. **PyTorch-compatible API** — code written against torch works unchanged

---

## The dtype problem

NumPy infers dtype from the input data. This is convenient but has two failure modes.

### Float64 is too wide

```python
np.array([1.0, 2.0, 3.0]).dtype   # float64
torch.tensor([1.0, 2.0, 3.0]).dtype  # float32
```

PyTorch uses `float32` by default because it is the standard training dtype —
it is half the memory of `float64` and hardware (GPUs, SIMD) is optimised for it.
If trainadillo produced `float64` tensors from Python float lists, a matrix
multiplication with a `float32` parameter would fail with a dtype mismatch.

**The fix:** After `np.array(data)`, if the inferred dtype is `float64` and no
`dtype` was explicitly provided, cast to `float32`.

```python
# tensor([1.0, 2.0]) without this fix:
#   → float64 (numpy default)
# tensor([1.0, 2.0]) with this fix:
#   → float32 (PyTorch default)
```

### Platform integer width

```python
np.arange(5).dtype   # int64 on Linux/macOS, int32 on Windows 64-bit
```

Token indices are used in matrix indexing. On Windows, `int32` indices into a large
vocabulary silently truncate. The fix is to always pass `dtype=np.int64` explicitly.

```python
def arange(n: int) -> Tensor:
    return Tensor(np.arange(n, dtype=np.int64))   # never platform int
```

---

## `tensor()` always copies

PyTorch's documentation states:

> `torch.tensor()` always copies data.

This is a deliberate safety guarantee. Consider what happens without copying:

```python
data = [1.0, 2.0, 3.0]
t = tensor(data)   # if this aliases data...
data[0] = 99.0     # ...this would silently corrupt t
```

Trainadillo matches PyTorch: `tensor()` always produces an independent copy.
If you need a non-copying path (e.g. wrapping a numpy array you already own),
`as_tensor()` would be the right function — but it's not needed yet.

```
torch.tensor()   →  always copies  →  safe, slightly slower
torch.as_tensor()  →  may alias    →  fast, requires care
```

Trainadillo only implements the safe default.

---

## `equal()` is not `==`

This is easy to get wrong. Trainadillo has two "equality" operations:

| Operation     | Returns            | Semantics                            |
| ------------- | ------------------ | ------------------------------------ |
| `a == b`      | `Tensor` (boolean) | Element-wise: which positions match? |
| `equal(a, b)` | `bool` (Python)    | Reduction: do all elements match?    |

```python
a = tensor([1, 2, 3])
b = tensor([1, 9, 3])

a == b          # → Tensor([True, False, True])
equal(a, b)     # → False
```

`equal()` is `torch.equal` — a shorthand for "same shape AND all elements identical."
It's useful for assertions in tests or checkpointing. The element-wise `==` is used
for masking (e.g. find all positions where token == padding_id).

Both are necessary. They're not duplicates.

---

## The `*_like` pattern

`zeros_like` and `full_like` create tensors that match the shape and dtype of an
existing tensor. This comes up often in autograd:

```python
grad = zeros_like(weight)    # gradient accumulator, same shape and dtype as weight
```

In PyTorch, the `_like` functions mirror the source tensor's dtype exactly. Trainadillo
delegates to numpy's equivalents, which do the same:

```python
def zeros_like(t: Tensor) -> Tensor:
    return Tensor(np.zeros_like(t._data))   # np.zeros_like preserves dtype
```

No special-casing needed — numpy handles it.

---

## Mirroring PyTorch's API

The creation API is designed so that `import trainadillo as torch` and then using
`torch.zeros`, `torch.tensor`, etc. works without changes to call sites:

```python
import trainadillo as torch   # (wired in T18)

t = torch.zeros(3, 4)         # (3,4) float32
ids = torch.arange(vocab_size)  # int64
batch = torch.stack(samples)    # new axis at dim 0
```

This mirroring strategy — PyTorch's API, NumPy's implementation — is the whole
design philosophy of trainadillo. The goal is not to be faster or more correct than
PyTorch; it's to be transparent enough that you can read the code and understand
exactly what PyTorch is doing at a high level.
