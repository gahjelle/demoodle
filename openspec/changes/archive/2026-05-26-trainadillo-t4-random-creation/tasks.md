## 1. Implementation

- [x] 1.1 Create `src/trainadillo/_random.py` with `rand(*shape, generator=None)` returning a float32 Tensor
- [x] 1.2 Add `randint(low, high, size, *, generator=None)` to `_random.py` returning an int64 Tensor
- [x] 1.3 Raise `RuntimeError` with a clear message when `generator=None` and `_default_generator is None`

## 2. Wiring

- [x] 2.1 Export `rand` and `randint` from `src/trainadillo/__init__.py`

## 3. Tests

- [x] 3.1 Create `tests/trainadillo/test_random.py`
- [x] 3.2 Test `rand` shape, dtype, value range, and reproducibility (same seed → same output, different seeds → different output)
- [x] 3.3 Test `randint` shape, dtype, value range, and reproducibility
- [x] 3.4 Test that calling `rand`/`randint` with no explicit generator and no `manual_seed` raises `RuntimeError`
- [x] 3.5 Test that an explicit `Generator` bypasses the default (works even before `manual_seed`)

## 4. Verification

- [x] 4.1 `uv run ruff format src/trainadillo/_random.py tests/trainadillo/test_random.py`
- [x] 4.2 `uv run ruff check src/trainadillo/_random.py tests/trainadillo/test_random.py`
- [x] 4.3 `uv run ty check src/ tests/`
- [x] 4.4 `uv run pytest tests/trainadillo/test_random.py`

## 5. Documentation

- [x] 5.1 Mark T4 as ✅ in `PLANS_TRAINADILLO.md`
- [x] 5.2 Write `docs/trainadillo/004-random-creation.md` explaining the design decisions (seeding contract, raise-on-unseeded, `size`-as-tuple)
