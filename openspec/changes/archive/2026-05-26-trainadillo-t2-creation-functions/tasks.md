## 1. Core Implementation

- [x] 1.1 Create `src/trainadillo/_creation.py` with `tensor()`, applying float64→float32 downgrade when dtype is not specified
- [x] 1.2 Add `zeros(*shape)` and `ones(*shape)` with explicit `dtype=np.float32`
- [x] 1.3 Add `zeros_like(t)` and `full_like(t, value)` delegating to `np.zeros_like` / `np.full_like`
- [x] 1.4 Add `arange(n)` with explicit `dtype=np.int64`
- [x] 1.5 Add `stack(tensors, dim=0)` delegating to `np.stack`
- [x] 1.6 Add `equal(a, b)` returning `bool` via `np.array_equal`

## 2. Tests

- [x] 2.1 Create `tests/trainadillo/test_creation.py`
- [x] 2.2 Test `tensor()`: ints → int64, floats → float32, explicit dtype override
- [x] 2.3 Test `tensor(existing_tensor)` copies (mutate original, check copy is unchanged)
- [x] 2.4 Test `zeros` / `ones` produce float32
- [x] 2.5 Test `zeros_like` / `full_like` preserve dtype and shape
- [x] 2.6 Test `arange` dtype is int64
- [x] 2.7 Test `stack` along dim=0 and dim=1
- [x] 2.8 Test `equal` returns True, False, and False for shape mismatch
- [x] 2.9 Test `tokens[offsets + context_len]` indexing pattern

## 3. Educational Docs

- [x] 3.1 Create `docs/trainadillo/002-creation-functions.md` explaining factory API design, the float64→float32 decision, `equal` vs `__eq__`, and the PyTorch mirroring strategy

## 4. Verification and Docs

- [x] 4.1 Run `uv run ruff format src/trainadillo/_creation.py tests/trainadillo/test_creation.py`
- [x] 4.2 Run `uv run ruff check src/trainadillo/_creation.py tests/trainadillo/test_creation.py`
- [x] 4.3 Run `uv run ty check src/ tests/`
- [x] 4.4 Run `uv run pytest tests/trainadillo/` — all tests pass
- [x] 4.5 Mark T2 done (✅) in `PLANS_TRAINADILLO.md`
- [x] 4.6 Review `CONTEXT.md` — add or update glossary entries if needed
