## Context

T1 delivered the `Tensor` class — a thin NumPy wrapper with no factory functions.
All existing tests construct tensors directly via `Tensor(np.array(...))`. Downstream
code (model layers, training loops, sampling) needs a PyTorch-compatible creation API
so it can write `tensor([1, 2, 3])` or `zeros(3, 4)` instead of reaching into NumPy.

The design constraint throughout is: **match PyTorch's behaviour wherever there is a
choice**, and **never depend on platform-specific NumPy defaults** (e.g. platform int
width, default float width).

## Goals / Non-Goals

**Goals:**
- Eight factory functions with PyTorch-compatible signatures and dtype behaviour
- No platform-dependent dtypes in any output
- `tensor()` always copies input data
- All functions return `Tensor` objects (except `equal`, which returns `bool`)

**Non-Goals:**
- Random factories (`rand`, `randint`) — depend on T3's `Generator`, covered in T4
- `as_tensor()` non-copying variant — not needed in the current codebase
- `torch.from_numpy()` — NumPy is the internal rep; no wrapping needed
- Wiring into `trainadillo/__init__.py` — deferred to T18

## Decisions

### D1: `tensor()` always copies

**Decision:** `np.array(data, ...)` (always copies) rather than `np.asarray(data, ...)`
(no-copy when input is already the right dtype/order).

**Rationale:** Matches PyTorch's documented guarantee. Prevents action-at-a-distance
bugs where mutating the source list or array silently changes the tensor.

**Alternative considered:** `np.asarray` with a flag — adds complexity for no benefit
at this scale.

### D2: Python float lists → float32, not float64

**Decision:** When `tensor()` infers dtype from a Python `float` or list of `float`s
and no explicit `dtype` is given, downgrade `float64` → `float32`.

**Rationale:** NumPy infers `float64` for Python floats; PyTorch defaults to `float32`.
Training code written against PyTorch expects `float32` tensors. Diverging here would
cause shape/dtype mismatches in matrix multiplications.

**Implementation:** After `np.array(data)`, if resulting dtype is `float64` and no
`dtype` was specified, cast to `float32`.

**Alternative considered:** Expose a `set_default_dtype()` function — premature for
this scope; the single rule (float64→float32) covers all current use cases.

### D3: `arange` uses explicit `np.int64`

**Decision:** Pass `dtype=np.int64` explicitly to `np.arange`.

**Rationale:** NumPy's default for integer `arange` is the platform C `long`, which is
`int32` on Windows 64-bit. Token indices must be `int64` everywhere to avoid silent
truncation. This matches `torch.arange(n)` for integer arguments.

### D4: `equal()` returns Python `bool`, not `Tensor`

**Decision:** `equal(a, b)` → `bool` via `np.array_equal`.

**Rationale:** This is `torch.equal` semantics — a reduction to a single truth value.
The element-wise `==` is already `Tensor.__eq__`, which returns a boolean `Tensor`.
The two operations are complementary, not duplicates, and must not be confused.

### D5: `zeros` / `ones` default to `float32`

**Decision:** Pass `dtype=np.float32` explicitly.

**Rationale:** NumPy defaults to `float64`; PyTorch defaults to `float32`. Training
parameters and activations are `float32`.

### D6: `zeros_like` / `full_like` inherit dtype from input

**Decision:** Delegate to `np.zeros_like` / `np.full_like`, which preserve dtype.

**Rationale:** Exactly matches PyTorch behaviour. No special-casing needed.

## Risks / Trade-offs

- **float64 → float32 silently narrows precision** → Acceptable: matches PyTorch and
  is the expected behaviour. Users who need `float64` can pass `dtype=float64` explicitly.
- **`tensor()` copies are O(n)** → Acceptable for training-scale tensors; copying on
  creation is the safe default.
- **`equal()` name shadows `Tensor.__eq__`** → Mitigated by keeping them on different
  namespaces: `equal(a, b)` vs `a == b`. The educational docs will call this out explicitly.

## Open Questions

None. All design decisions are resolved.
