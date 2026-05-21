## Why

The project has a config layer and a CLI skeleton, but no shared value types — there is no common vocabulary for what flows between stages (corpus, tokenized data, model weights, metrics). Without these types, nothing in the pipeline can be wired together.

## What Changes

- Add `src/demoodle/core/types.py` with all core value types
- Add `tests/test_core_types.py` verifying each type is constructable and frozen
- Add `docs/architectures/` with three architecture explainers (bigram, MLP, transformer)
- Add `agents/docs-practices.md` describing conventions for writing notes in `docs/`

## Capabilities

### New Capabilities

- `core-value-types`: The immutable data spine — `Seq`, `Output`, `Corpus`, `Tokenizer`, `Dataset`, `Policy`, `Metrics`, and the `Artifact` union
- `architecture-docs`: Human-readable explainers for each model architecture, aimed at Python-proficient readers new to LLMs

### Modified Capabilities

## Impact

- `src/demoodle/core/types.py` — new file, no existing code changes
- `tests/test_core_types.py` — new test file
- `docs/architectures/` — new directory with three markdown files
- `agents/docs-practices.md` — new instructions file
- No breaking changes; no existing imports affected
- PyTorch (`torch`) is already a dependency; no new packages needed

## Non-goals

- Implementing any behavior (encoding, training, inference) — that belongs to later work items
- Defining protocols or behavioral contracts — those come in W4
- Persistence or serialization of artifacts — that is W5
