## 1. Extract Shared Sampling

- [x] 1.1 Create `src/demoodle/architectures/sampling.py` with the `sample` function (extracted from `bigram.py`, underscore removed)
- [x] 1.2 Update `src/demoodle/architectures/bigram.py` to import `sample` from `architectures.sampling` and remove the local `_sample` definition
- [x] 1.3 Update `tests/architectures/test_bigram.py` to import `sample` from `demoodle.architectures.sampling` instead of `_sample` from `demoodle.architectures.bigram`
- [x] 1.4 Run `uv run pytest tests/architectures/test_bigram.py` — all existing bigram tests must pass

## 2. Trigram Model and Architecture

- [x] 2.1 Create `src/demoodle/architectures/trigram.py` with `TrigramModel` (V×V×V `nn.Parameter`, `forward` handles shape `(2,)` and `(batch, 2)`) and `TrigramArchitecture` (`context_length=2`, `init_state`, `forward`, `call`, `explain`)
- [x] 2.2 Write `tests/architectures/test_trigram.py` covering: model shape, row lookup (single + batched), parameter count, `init_state` determinism, `context_length == 2`, `forward` uses last 2 tokens, `call` returns valid `sampled_ids`, temperature/top_k/top_p behaviour, `explain` returns `{}`

## 3. Config and CLI Wiring

- [x] 3.1 Add `TrigramConfig` (empty, like `BigramConfig`) to `src/demoodle/config/schemas.py` and add `trigram: TrigramConfig` field to `ArchitecturesConfig`
- [x] 3.2 Add `[architecture.trigram]` section to `src/demoodle/config/demoodle.toml`
- [x] 3.3 Add `case "trigram"` to `_make_arch` in `src/demoodle/frontends/cli.py` returning `TrigramArchitecture(vocab_size=vocab_size)`
- [x] 3.4 Update `_generate` in `src/demoodle/frontends/cli.py` to pad sequences shorter than `context_length` with the `\n` token (`tokenizer.encode("\n")[0]`) before calling `arch.call`

## 4. Tests

- [x] 4.1 Run `uv run pytest tests/architectures/test_trigram.py` — all trigram tests pass
- [x] 4.2 Run `uv run pytest` — full suite passes

## 5. Documentation

- [x] 5.1 Create `docs/architectures/trigram.md` — educational doc explaining V×V×V lookup, training, generation, the V³ scaling limitation, and how trigram sits between bigram and MLP
- [x] 5.2 Update `docs/architectures/bigram.md` — replace the "Relation to MLP" section with a "Relation to Trigram and MLP" section placing trigram as the intermediate step
- [x] 5.3 Update `CONTEXT.md` — in the `ArchitectureProtocol` definition, extend the `context_length` description to note `2 for trigram` alongside `1 for bigram`
- [x] 5.4 Update `src/demoodle/architectures/__init__.py` docstring to include trigram in the listed architectures
- [x] 5.5 Update `PLANS.md` W13 `Done when` — change "lower loss / better samples than bigram" to include trigram as a comparison point
- [x] 5.6 Update `PLANS.md` W17 `Build` — add trigram to "bigram & MLP still return `{}`"

## 6. Verification

- [x] 6.1 Run `uv run ruff format src/ tests/` and `uv run ruff check src/ tests/`
- [x] 6.2 Run `uv run ty check src/ tests/`
- [x] 6.3 Run `uv run pytest` — full suite green
- [x] 6.4 Switch `architecture.active = "trigram"` in `demoodle.toml`, run `uv run demoodle train` and `uv run demoodle call` — training loss decreases and names are generated
- [x] 6.5 Restore `architecture.active = "bigram"` and confirm bigram still works unchanged
