## 1. Tests (red)

- [x] 1.1 Create `tests/core/test_rng.py` with failing tests for: immutability, split determinism, split children diverge, different parents produce different children, generator reproducibility, different seeds produce different draws

## 2. Implementation (green)

- [x] 2.1 Create `src/demoodle/core/rng.py` with frozen `RNG` dataclass, `_mix` helper using `hashlib.sha256`, `.split() -> tuple[RNG, RNG]`, and `.generator() -> torch.Generator`
- [x] 2.2 Export `RNG` from `src/demoodle/core/__init__.py`

## 3. Verification

- [x] 3.1 Run `uv run pytest tests/core/test_rng.py` — all tests pass
- [x] 3.2 Run `uv run ruff format src/ tests/` and `uv run ruff check src/ tests/`
- [x] 3.3 Run `uv run ty check src/ tests/`
- [x] 3.4 Run full test suite: `uv run pytest`

## 4. Documentation

- [x] 4.1 Read `PLANS.md` and mark W3 as ✅ done
- [x] 4.2 Read files in `agents/` and update any relevant documentation
