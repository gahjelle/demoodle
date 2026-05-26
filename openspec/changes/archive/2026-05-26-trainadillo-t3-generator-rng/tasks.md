## 1. Core Implementation

- [x] 1.1 Create `src/trainadillo/_rng.py` with module-level
  `_default_generator: Generator | None = None`
- [x] 1.2 Implement `Generator.__init__`: seed from OS entropy via
  `numpy.random.default_rng()`; store as `_np_rng`; set `_seeded = False`
- [x] 1.3 Implement `Generator.manual_seed(seed: int) -> Generator`: reinitialize
  `_np_rng` with `numpy.random.Generator(numpy.random.PCG64(seed))`; set
  `_seeded = True`; return `self`
- [x] 1.4 Implement `Generator.__repr__`: return `"Generator(seeded)"` or
  `"Generator(unseeded)"` based on `_seeded`
- [x] 1.5 Implement module-level `manual_seed(seed: int) -> Generator`: seed
  `_default_generator` (creating a new `Generator` if `None`); return it

## 2. Tests

- [x] 2.1 Create `tests/trainadillo/test_rng.py`
- [x] 2.2 Test `Generator()` constructs without error and `_np_rng` is a
  `numpy.random.Generator`
- [x] 2.3 Test same seed reproduces the same sequence of raw draws from `_np_rng`
- [x] 2.4 Test different seeds produce different sequences
- [x] 2.5 Test `generator.manual_seed(42)` returns `self`
- [x] 2.6 Test `repr(Generator())` is `"Generator(unseeded)"`
- [x] 2.7 Test `repr(g)` after `g.manual_seed(42)` is `"Generator(seeded)"`
- [x] 2.8 Test `trainadillo.manual_seed(42)` returns a `Generator` with
  `repr == "Generator(seeded)"`
- [x] 2.9 Test `trainadillo.manual_seed(42)` called twice with the same seed produces
  generators whose subsequent draws match
- [x] 2.10 Test `trainadillo._rng._default_generator is None` in a fresh module state
  (import the module fresh — use `importlib.reload` or a subprocess)
- [x] 2.11 Test `trainadillo._rng._default_generator is not None` after
  `trainadillo.manual_seed(42)`

## 3. Educational Doc

- [x] 3.1 Create `docs/trainadillo/003-rng.md` covering:
  - Why ML needs reproducible randomness: debugging, ablations, paper reproduction —
    "my loss went up" is only diagnosable if you can reproduce it
  - Global mutable state (PyTorch) vs explicit keys (JAX): the two schools, their
    tradeoffs, why JAX's approach is stricter but harder to misuse
  - PCG64 vs Mersenne Twister vs Philox: statistical quality, speed, GPU parallelism;
    why PCG64 is the right choice for a CPU-only NumPy project
  - Why trainadillo seeds don't reproduce PyTorch sequences (same seed, different
    algorithm → different numbers; reproducibility is within-framework)
  - The module-level footgun: why trainadillo makes unseeded `rand()` loud, and how
    this maps to the JAX philosophy applied surgically to one path
  - `Generator(seeded=False)` repr as a debugging signal: what it means, when you'd
    see it, what to do about it
  - How Demoodle uses this: always explicit seeds via `RNG.generator()` (JAX
    discipline with a PyTorch API shape), so the module-level default is never
    reached in production code

## 4. Verification

- [x] 4.1 `uv run ruff format src/trainadillo/_rng.py tests/trainadillo/test_rng.py`
- [x] 4.2 `uv run ruff check src/trainadillo/_rng.py tests/trainadillo/test_rng.py`
- [x] 4.3 `uv run ty check src/ tests/`
- [x] 4.4 `uv run pytest tests/trainadillo/`

## 5. Documentation

- [x] 5.1 Mark T3 done (✅) in `PLANS_TRAINADILLO.md`
- [x] 5.2 Review `CONTEXT.md` — add or update glossary entries if needed (e.g.
  `trainadillo.Generator` alongside existing `RNG` entry)
