# T1 — Tensor Class & Dtype Constants

## The concept: why wrap numpy at all?

A `numpy.ndarray` is just a block of numbers in memory with a shape and a dtype.
It is very fast, but it has no concept of *where it came from*. If you do
`c = a + b` in numpy, `c` knows nothing about `a` or `b` — it cannot tell you
"I was created by adding these two arrays."

PyTorch's `Tensor` is a wrapper that adds this memory. When gradient tracking is
enabled, every operation records a *grad function* — a closure that knows how to
compute the gradient of the output with respect to each input. `Tensor.backward()`
walks those closures in reverse and accumulates gradients into the leaf tensors.

T1 builds just the wrapper. The memory — the autograd fields — arrives in T8.
Think of T1 as laying the foundation of a building; T8 wires the electricity.

```
┌──────────────────────────────────────────────────────┐
│  trainadillo.Tensor  (T1 builds this)                │
│                                                      │
│  _data: np.ndarray  ← the actual numbers             │
│  shape, ndim, dtype ← metadata about the numbers    │
│  +, -, *, /, @      ← non-differentiable for now    │
│  __getitem__        ← indexing                       │
│                                                      │
│  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌  │
│  grad               ← T8 adds these                  │
│  requires_grad      │                                │
│  _grad_fn           │                                │
│  _is_leaf           ←                                │
└──────────────────────────────────────────────────────┘
```

## `Size` — why not just return a tuple?

`Tensor.shape` returns a `Size`, not a plain tuple. `Size` is a tuple subclass
with two differences from a plain tuple:

1. **`numel()`** — returns the product of all dimensions. `Size([3, 4]).numel()` is
   `12`. You could write `math.prod(tensor.shape)` instead, but `tensor.size().numel()`
   is idiomatic in PyTorch code.

2. **`repr`** — `repr(tensor.shape)` prints `torch.Size([3, 4])`, not `(3, 4)`. This
   matches PyTorch output exactly, which matters when reading tutorials or comparing
   output to real PyTorch.

The name `torch.Size` in the repr is intentional — even though this is trainadillo,
the repr is chosen to match PyTorch's output for easy side-by-side comparison.

Real PyTorch's `torch.Size` is defined in C++ (`THPSizeType`). Our implementation
is a pure Python tuple subclass that achieves the same interface.

## Why `__bool__` raises TypeError

If you write `if tensor:` in Python, Python calls `__bool__()` on the object. For
a weight matrix of shape `(27, 27)` — which element's truthiness would that mean?
The first one? The sum? There is no sensible answer for a multi-element tensor.

PyTorch raises `RuntimeError: Boolean value of Tensor with more than one element is
ambiguous`. Trainadillo raises `TypeError` for the same reason (TypeError is the
appropriate Python error when a conversion is not supported).

The practical effect: this forces you to be explicit:
- **For a scalar:** `t.item()` extracts a Python scalar, which you can use in `if`
- **For an array:** `t.any()` or `t.all()` (added in T6/T7)

There is a second reason: boolean Tensors used as *masks* must not accidentally
collapse during expression evaluation. `(cumulative - sorted_probs) > top_p`
produces a boolean Tensor — if that Tensor could silently become a Python bool,
the masking logic in `_sample()` would break.

## Why integer indexing returns a 0-D Tensor, not a Python scalar

```python
t = Tensor(np.array([10.0, 20.0, 30.0]))
t[1]        # → Tensor with shape () and value 20.0
t[1].item() # → 20.0  (Python float)
```

NumPy itself returns a scalar: `np.array([1, 2, 3])[1]` is `np.int64(2)`, not an
array. We explicitly wrap with `np.asarray()` to force a 0-D array, then wrap that
in a `Tensor`.

Why does this matter? T11 adds autograd to indexing — `weight[token_id]` needs to
remain inside the computation graph so that gradients flow back to `weight`. If the
result were a Python float, it would escape the graph entirely and `weight.grad`
would never be populated.

So the rule is: **indexing a Tensor always produces a Tensor**. To get a Python
scalar, you must explicitly call `.item()`.

## `view()` — one method, two behaviours

`view()` is one of PyTorch's quirkier design decisions. It handles two completely
different operations depending on the argument type:

```python
tensor.view(3, 4)      # reshape: same data, new shape
tensor.view(torch.uint8)  # reinterpret: same bytes, new dtype
```

The reshape form is like `numpy.reshape`. The reinterpret form is like
`numpy.ndarray.view(dtype)` — it reinterprets the raw bytes as a different type.
A float32 tensor of 3 elements becomes a uint8 tensor of 12 elements (4 bytes each).

Trainadillo dispatches at runtime by checking whether the first argument is a numpy
dtype type (`issubclass(arg, np.generic)`). If yes, it's a dtype reinterpret; if
no, it's a reshape. This dual behaviour is needed because `persistence.py` uses
`.view(torch.uint8)` to convert float weights into raw bytes for hashing.

## Dtype constants — `long`, `uint8`, `float32`

```python
import trainadillo as torch

torch.long    # → np.int64
torch.float32 # → np.float32
torch.uint8   # → np.uint8
```

These are the numpy scalar type classes themselves. Passing them as `dtype=` to any
numpy function works directly. In real PyTorch, `torch.long` is a `torch.dtype`
enum value that maps to C's `int64_t`. Our numpy types play the same role but
without the C layer.

The constants are intentionally lowercase to match PyTorch's convention (`torch.long`,
not `torch.LONG`). They're not ALL_CAPS module-level constants in the traditional
Python sense — they're type aliases.

## Reflected arithmetic operators

```python
2.0 * tensor  # calls tensor.__rmul__(2.0)
1.0 / tensor  # calls tensor.__rtruediv__(1.0)
```

When Python evaluates `2.0 * tensor`, it first tries `float.__mul__(2.0, tensor)`.
The float doesn't know how to multiply by a Tensor, so it returns `NotImplemented`.
Python then tries the reflected operator: `tensor.__rmul__(2.0)`. This is why we
implement both `__mul__` and `__rmul__` — the forward version handles `tensor * 2.0`,
the reflected version handles `2.0 * tensor`.

## Tensors are unhashable (`__hash__ = None`)

Setting `__hash__ = None` means `Tensor` instances cannot be used in sets or as
dict keys. PyTorch Tensors are also unhashable by default (you'd get `TypeError`).
This is intentional: two tensors with the same values are not "the same" — they
may have different gradient histories. Hashing by identity (the default) would be
misleading. Hashing by value would be expensive and semantically wrong.

## What T8 adds on top of this foundation

T8 extends `Tensor` with four autograd fields:

| Field | Type | Purpose |
|---|---|---|
| `grad` | `Tensor \| None` | Accumulated gradient, populated by `.backward()` |
| `requires_grad` | `bool` | Whether this tensor participates in gradient tracking |
| `_grad_fn` | `GradFn \| None` | The function that created this tensor (None for leaves) |
| `_is_leaf` | `bool` | True for tensors created by the user; False for intermediate results |

T10 then upgrades the arithmetic dunders (currently dumb numpy delegates) to check
`requires_grad` and attach `GradFn` nodes when building the computation graph.
