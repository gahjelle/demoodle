## Context

T3 is T4's dependency: `rand()` and `randint()` need a controllable source of
randomness. The `Generator` class provides that — a thin wrapper around
`numpy.random.Generator` (PCG64) with PyTorch's public interface.

The module-level default is where the design deliberately diverges from PyTorch.
PyTorch seeds its default generator with OS entropy at import time, so calling
`rand()` without seeding silently produces non-reproducible results. Trainadillo
treats the module-level default as unset until `manual_seed()` is called — making
the unseeded path a loud error. Demoodle always passes explicit seeds (JAX-style
discipline), so the expected use case is unaffected.

## Goals / Non-Goals

**Goals:**
- `Generator` wraps numpy's PCG64 with PyTorch's public interface
- `Generator()` uses OS entropy and is immediately usable — explicit creation is
  a conscious choice
- Module-level `manual_seed(seed)` seeds the default and returns it
- Module-level default is `None` until seeded; T4 raises if it encounters `None`
- `Generator.__repr__` shows seeded state for easy debugging
- T4 accesses `generator._np_rng` directly to call numpy random methods

**Non-Goals:**
- `rand`, `randint` (T4)
- `initial_seed()`, `get_state()`, `set_state()`
- `__init__.py` wiring (T18)

## Decisions

### D1: PCG64 backing

**Decision:** Use `numpy.random.Generator(numpy.random.PCG64(seed))` internally.

**Rationale:** PCG64 is numpy's recommended PRNG since numpy 1.17. It has better
statistical properties than MT19937 (Mersenne Twister) and is faster. PyTorch uses
Philox for GPU-friendly parallel draws — trainadillo has no GPU requirement, so
PCG64 is the right choice.

**Consequence:** trainadillo seeds do not reproduce PyTorch sequences. Same seed,
different algorithm → different numbers. Reproducibility is within-trainadillo only.
The educational doc will call this out explicitly.

### D2: `Generator()` uses OS entropy; immediately usable

**Decision:** `Generator()` with no arguments seeds from OS entropy via
`numpy.random.default_rng()`. The generator is valid and usable immediately —
no call to `manual_seed()` required.

**Rationale:** When you explicitly write `g = Generator()`, you are consciously
creating an RNG object. OS entropy is the right default for a user who wants a
generator but doesn't need a specific seed. This matches PyTorch's `torch.Generator()`
behaviour exactly, preserving full compatibility for the explicit-generator path.

**Alternative rejected:** Raise on first use unless seeded. This would make
`Generator()` unusable without `manual_seed()` — an unnecessary departure from
PyTorch for the explicit case. The footgun lives specifically in the module-level
default path, not in explicit generator creation.

### D3: Module-level `_default_generator` starts as `None`; T4 raises if `None`

**Decision:** `_default_generator: Generator | None = None` at module load. T4's
`rand(generator=None)` and `randint(generator=None)` check: if the resolved
generator is `None`, raise:

```python
raise RuntimeError(
    "trainadillo: random op called without a generator and no default seed is set. "
    "Call trainadillo.manual_seed(seed) before using random functions, "
    "or pass generator= explicitly."
)
```

**Rationale:** The module-level default is the footgun path — you didn't think
about randomness, you just called `rand()`. A loud error here is better than silent
non-reproducibility. It is also testable: a test can confirm that all Demoodle
code paths seed before drawing random numbers.

**Asymmetry from D2:** `Generator()` (explicit creation) gets OS entropy;
`rand()` without a seed raises. The asymmetry maps to awareness at the call site:
explicit creation is a conscious choice; reaching the module-level default unseeded
is a likely oversight.

### D4: Both `manual_seed` forms return the seeded `Generator`

**Decision:**
- `generator.manual_seed(seed: int) -> Generator` returns `self`
- `trainadillo.manual_seed(seed: int) -> Generator` returns the now-seeded
  `_default_generator`

**Rationale:** Matches PyTorch: `torch.manual_seed(42)` returns the Generator.
Enables chaining (`g = trainadillo.manual_seed(42)`) and nudges toward the explicit
generator style — the returned `g` can be passed to `rand(generator=g)` directly.

### D5: `Generator.__repr__` shows seeded state

**Decision:**
```python
def __repr__(self) -> str:
    state = "seeded" if self._seeded else "unseeded"
    return f"Generator({state})"
```

`_seeded` is `False` for OS-entropy generators (explicit `Generator()` without
`manual_seed`), `True` after any `manual_seed()` call.

**Rationale:** Makes RNG state inspectable at a glance. A `Generator(unseeded)`
in a repr or debug print immediately tells you why results aren't reproducible.
Using plain words rather than `seeded=True/False` avoids the misreading that
`seeded` is a constructor parameter.

### D6: T4 accesses `_np_rng` directly

**Decision:** T4's `rand()` and `randint()` call `generator._np_rng.random(shape)`
and `generator._np_rng.integers(low, high, size)`. `_np_rng` is a semi-private
attribute (underscore = internal plumbing, not part of the public API).

**Rationale:** Avoids reimplementing numpy's entire random API on `Generator`.
Within trainadillo itself this is acceptable; external callers should use `rand()`
and `randint()`, not `_np_rng` directly.

## Risks / Trade-offs

- **Seeds don't reproduce PyTorch sequences** → Documented in the educational doc.
  Not a correctness issue; reproducibility is within-trainadillo.
- **Module-level default diverges from PyTorch** → Intentional; trainadillo is
  stricter. Code that explicitly seeds will work identically with PyTorch.
- **`Generator(unseeded)` repr for OS-entropy generators could be read as "broken"** →
  Acceptable. The signal is "you haven't explicitly controlled this seed", which
  is accurate. The educational doc explains the distinction.

## Open Questions

None. All design decisions resolved during exploration.
