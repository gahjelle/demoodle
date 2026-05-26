# T4 — Random Creation Functions

**Concepts introduced:** `rand`, `randint`, seeding contract, fail-loud reproducibility, PyTorch's `size`-as-tuple signature

---

## What we built

Two random tensor factories that mirror PyTorch's API:

```python
import trainadillo as torch

torch.manual_seed(42)

x = torch.rand(3, 4)          # shape (3, 4), float32, values in [0, 1)
y = torch.randint(0, 10, (5,)) # shape (5,), int64, values in [0, 10)
```

Or with an explicit generator for finer control:

```python
g = torch.Generator()
g.manual_seed(99)
tokens = torch.randint(0, vocab_size, (batch,), generator=g)
```

---

## Why a separate `_random.py`?

The earlier `_creation.py` (T2) contains deterministic factories — `zeros`, `ones`, `tensor`, etc. — with no dependency on the RNG. Adding `rand` there would couple two unrelated concerns and force `_creation.py` to import `_rng.py`.

A separate `_random.py` keeps the boundary clean: deterministic construction lives in one place, stochastic construction in another. This mirrors how PyTorch's C++ source separates `torch/csrc/autograd/` (core ops) from `torch/csrc/Generator.cpp` (RNG state).

---

## The seeding contract: loud failure over silent non-reproducibility

PyTorch uses OS entropy as a fallback when you haven't called `torch.manual_seed()`:

```python
# PyTorch: silently non-reproducible
x = torch.rand(3)  # different every run — no warning
```

Trainadillo is more strict. The module-level `_default_generator` starts as `None`:

```python
# _rng.py
_default_generator: Generator | None = None
```

Calling `rand()` or `randint()` without a seed raises immediately:

```python
trainadillo.rand(3)
# RuntimeError: No random generator available.
# Call `trainadillo.manual_seed(seed)` before using `rand` or `randint`,
# or pass an explicit `generator=` argument.
```

**Why?** Trainadillo is a learning project. Reproducibility is a first-class goal. Silent non-determinism makes experiments hard to reason about and debug. PyTorch's silent-entropy fallback is pragmatic for production code but teaches a bad habit. Loud failure here forces intentional seeding — and that habit carries over when you're writing real PyTorch training loops.

The design is also consistent with T3's `Generator`: every `Generator()` starts from OS entropy (valid immediately), but the *module default* is explicitly unset. You choose when module-level randomness becomes reproducible.

---

## `size` is a tuple, not `*args`

A common PyTorch gotcha:

```python
# PyTorch / Trainadillo — size is a TUPLE:
torch.randint(0, 10, (5,))   # ✓  returns shape-(5,) tensor
torch.randint(0, 10, 5)      # ✗  TypeError
```

Compare with `rand`, which uses `*shape`:

```python
torch.rand(3, 4)    # ✓  shape args unpacked
torch.rand((3, 4))  # ✗  shape would be ((3, 4),)
```

The inconsistency is historical: `rand` predates NumPy-style size tuples in PyTorch; `randint` was added later with a more consistent interface. Trainadillo matches both signatures so code targeting PyTorch works unchanged.

---

## Accessing `_np_rng` directly

The `Generator` class exposes `_np_rng` as semi-private plumbing. The `rand` and `randint` implementations call it directly:

```python
rng._np_rng.random(shape).astype(np.float32)   # rand
rng._np_rng.integers(low, high, size=size, dtype=np.int64)  # randint
```

This is intentional. `_np_rng` is a `numpy.random.Generator` — the modern NumPy RNG API (PCG64). Using it directly avoids an intermediate abstraction layer and makes the NumPy translation obvious. The underscore signals "internal plumbing" but not "never touch."

In real PyTorch, this layer is C++ (a `THGenerator` struct wrapping Philox RNG state). Trainadillo's direct Python/NumPy equivalent is as close as we can get without C extensions.

---

## dtype choices

| Function  | Output dtype | Why                                                                                           |
| --------- | ------------ | --------------------------------------------------------------------------------------------- |
| `rand`    | `float32`    | Matches PyTorch default dtype; NumPy's `random()` returns `float64`, so we `.astype(float32)` |
| `randint` | `int64`      | Matches PyTorch's `torch.int64`; avoids int32/int64 mismatch bugs on Windows                  |

---

## Connection to the rest of trainadillo

`rand` and `randint` are used primarily in the **sampling path** — not the training path. In `bigram.py`:

- `randint` would generate initial random token sequences for testing
- `rand` feeds into temperature-based sampling and top-p filtering

Neither function needs to be differentiable. Sampling is an inference operation; gradients don't flow through random token selection. (If you needed `rand` to be differentiable — e.g., for the reparameterization trick in variational autoencoders — you'd need to implement it as a differentiable op with a GradFn. That's a much later topic.)
