## Why

T2 delivered factory functions for deterministic tensors. T4 will add `rand` and
`randint` — random factories that need a controllable RNG state. T3 builds the
`Generator` class that T4 depends on: a thin wrapper around numpy's PCG64 generator
with PyTorch's `Generator` interface.

Trainadillo also needs a module-level `manual_seed()` matching `torch.manual_seed()`
so existing PyTorch idioms work. The design diverges from PyTorch in one deliberate
way: the module-level default generator is unset until `manual_seed()` is called,
making unseeded random ops a loud error rather than a silent source of
non-reproducibility. Demoodle intends to always pass explicit seeds (JAX discipline),
so the expected use case is fully compatible.

## What Changes

- **New file** `src/trainadillo/_rng.py`: `Generator` class and module-level
  `manual_seed()`
- **New file** `tests/trainadillo/test_rng.py`: unit tests for `Generator` and
  `manual_seed()`
- **New file** `docs/trainadillo/003-rng.md`: deep-dive educational doc on RNG
  concepts — PCG64 vs MT19937 vs Philox, global vs explicit generators, JAX vs
  PyTorch philosophies, why reproducibility matters in ML

## Capabilities

### New Capabilities

- `trainadillo-generator`: The `Generator` type and module-level RNG control —
  wrapping numpy's PCG64 with a PyTorch-compatible interface and an
  explicit-seeding-required module-level default.

## Impact

- `src/trainadillo/_rng.py` — new file
- `tests/trainadillo/test_rng.py` — new file
- `docs/trainadillo/003-rng.md` — new learning doc
- No changes to existing files
- No `demoodle.*` imports; trainadillo remains self-contained

## Non-goals

- Random creation functions (`rand`, `randint`) — T4; depend on T3's `Generator`
- `initial_seed()`, `get_state()`, `set_state()` — not needed for the current codebase
- `fork_rng()` context manager — not needed
- Per-device generators (CPU/CUDA) — trainadillo is CPU-only
- Wiring into `trainadillo/__init__.py` — deferred to T18
