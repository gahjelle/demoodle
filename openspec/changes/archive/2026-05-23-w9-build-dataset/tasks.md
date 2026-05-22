## 1. Stage implementation

- [x] 1.1 Create `src/demoodle/data/stages.py` with `make_build_dataset_stage(config_hash: str = "") -> Stage`
- [x] 1.2 Implement: normalize trailing `\n`, encode via `tokenizer.encode()`, return `Dataset(tokens=torch.tensor(..., dtype=torch.long))`

## 2. Tests

- [x] 2.1 Write failing tests in `tests/data/test_stages.py`: correct token sequence for known input, trailing `\n` added when missing, existing trailing `\n` not duplicated, tensor dtype is `torch.long`, tensor is 1D
- [x] 2.2 Write failing test: stage runs via the runner and produces `dataset` artifact in output dict
- [x] 2.3 Confirm all tests pass

## 3. Verification

- [x] 3.1 `uv run ruff format src/ tests/`
- [x] 3.2 `uv run ruff check src/ tests/`
- [x] 3.3 `uv run ty check src/ tests/`
- [x] 3.4 `uv run pytest`

## 4. Documentation

- [x] 4.1 Mark W9 done (✅) in `PLANS.md`
- [x] 4.2 Read `README.md` and update if necessary
- [x] 4.3 Review `CONTEXT.md` and add/update terms if necessary (dataset, token sequence, build_dataset stage)
- [x] 4.4 Read files inside `agents/` and update if necessary
