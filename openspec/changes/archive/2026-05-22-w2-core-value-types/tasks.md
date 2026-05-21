## 1. Core Value Types

- [x] 1.1 Implement `src/demoodle/core/types.py` with `Seq`, `Output`, `Corpus`, `Tokenizer`, `Dataset`, `Policy`, `Metrics`, and `Artifact`
- [x] 1.2 Add stub comments in `Artifact` for `RewardModel` and `PreferenceData` (Milestone 5)

## 2. Tests

- [x] 2.1 Write `tests/test_core_types.py` — construct one instance of each type and assert it is frozen (`FrozenInstanceError` on field assignment)
- [x] 2.2 Assert `Artifact` union includes all five variants

## 3. Architecture Documentation

- [x] 3.1 Create `docs/architectures/bigram.md` — TLDR + detailed explainer for the learned-bigram model
- [x] 3.2 Create `docs/architectures/mlp.md` — TLDR + detailed explainer for the Bengio-style MLP
- [x] 3.3 Create `docs/architectures/transformer.md` — TLDR + detailed explainer for the tiny transformer
- [x] 3.4 Add ASCII diagrams to each explainer; add SVG files to `docs/architectures/` for any diagram too complex for ASCII

## 4. Agent Instructions

- [x] 4.1 Create `agents/docs-practices.md` — conventions for writing files in `docs/` (audience, structure, tone, placement, diagram conventions: ASCII-first, SVG for complex diagrams)

## 5. Verification

- [x] 5.1 Run `uv run ruff format src/ tests/` and fix any issues
- [x] 5.2 Run `uv run ruff check src/ tests/` and fix any issues
- [x] 5.3 Run `uv run ty check src/ tests/` and fix any issues
- [x] 5.4 Run `uv run pytest` and confirm all tests pass

## 6. Documentation Review

- [x] 6.1 Read `PLANS.md` and mark W2 as ✅ done
- [x] 6.2 Read `README.md` and update if necessary
- [x] 6.3 Read files inside `agents/` and update if necessary
