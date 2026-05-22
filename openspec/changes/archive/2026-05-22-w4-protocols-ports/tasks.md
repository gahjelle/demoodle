## 1. Tests (red)

- [x] 1.1 Create `tests/ports/` package with `__init__.py`
- [x] 1.2 Create `tests/ports/test_protocols.py` with failing tests for: `TokenizerProtocol` dummy satisfies protocol, `ArchitectureProtocol` dummy satisfies protocol, `InspectableProtocol` dummy with only `call` satisfies protocol and `explain()` returns `{}`, `InspectableProtocol` with overridden `explain` returns custom data, `Stage` is frozen (raises `FrozenInstanceError` on mutation), all four names importable from `demoodle.ports`

## 2. Implementation (green)

- [x] 2.1 Implement `src/demoodle/ports/protocols.py`: `TokenizerProtocol` (`encode`, `decode`, `vocab_size`), `ArchitectureProtocol` (`init_state`, `forward`), `InspectableProtocol` (`call` required, `explain` with default `return {}`), `Stage` frozen dataclass (`name`, `needs`, `produces`, `run`)
- [x] 2.2 Export all four names from `src/demoodle/ports/__init__.py`

## 3. Verification

- [x] 3.1 Run `uv run pytest tests/ports/test_protocols.py` — all tests pass
- [x] 3.2 Run `uv run ruff format src/ tests/` and `uv run ruff check src/ tests/`
- [x] 3.3 Run `uv run ty check src/ tests/`
- [x] 3.4 Run full test suite: `uv run pytest`

## 4. Documentation

- [x] 4.1 Read `PLANS.md` and mark W4 as ✅ done
- [x] 4.2 Read files in `agents/` and update any relevant documentation
