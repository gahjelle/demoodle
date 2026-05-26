## Context

T1 (Tensor), T2 (creation functions), and T3 (Generator/RNG) are complete. The `Generator` class wraps NumPy's PCG64 and exposes `_np_rng` for direct use. A module-level `_default_generator` exists in `_rng.py` but starts as `None` — it is only set by `trainadillo.manual_seed()`.

T4 adds the two random creation functions that appear in the sampling path (`rand`, `randint`) and wires them to the existing RNG infrastructure.

## Goals / Non-Goals

**Goals:**
- `rand(*shape, generator=None)` producing float32 uniform [0, 1) Tensors
- `randint(low, high, size, *, generator=None)` producing int64 Tensors, `size` is a tuple
- Explicit error when no default generator has been seeded (reproducibility is opt-in, not silent)
- Exports from `trainadillo/__init__.py`

**Non-Goals:**
- `randn`, `normal`, `bernoulli`, or any other random op
- Changing `Generator` or `manual_seed`
- GPU or multi-device concerns

## Decisions

### New file `_random.py` rather than extending `_creation.py`

`_creation.py` is currently self-contained (no `_rng.py` import). Adding random functions there would couple deterministic and stochastic creation. A separate `_random.py` keeps that boundary clean and mirrors how `_rng.py`'s existing comment already anticipated separate consumers.

*Alternative considered:* keep everything in `_creation.py` for fewer files. Rejected because the import coupling outweighs the convenience.

### Raise on unseeded default generator

When `generator=None` and `_default_generator is None`, raise `RuntimeError`. This matches the comment already in `_rng.py` and makes reproducibility problems loud at call time rather than silent. Users must call `trainadillo.manual_seed(seed)` before using random ops without an explicit generator.

*Alternative considered:* auto-initialize from OS entropy (PyTorch behavior). Rejected — trainadillo is an educational project where reproducibility is a first-class constraint, and silently non-reproducible code defeats that purpose.

### `size` as a tuple in `randint`

PyTorch's `randint(low, high, size)` takes `size` as a positional tuple, not `*args`. This is a common PyTorch footgun; trainadillo matches it exactly so code written against PyTorch works unchanged.

## Risks / Trade-offs

- **Unseeded raise is a usability surprise** → Mitigation: error message should be explicit: "Call `trainadillo.manual_seed(seed)` before using `rand` or `randint` without an explicit generator."
- **`_np_rng` is semi-private** → Mitigation: `_rng.py` documents this as intentional plumbing for T4 consumers; no external callers expected.

## Open Questions

None.
