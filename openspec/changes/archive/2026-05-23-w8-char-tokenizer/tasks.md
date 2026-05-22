## 1. CharTokenizer implementation

- [x] 1.1 Create `src/demoodle/tokenizers/__init__.py`
- [x] 1.2 Write failing tests in `tests/tokenizers/test_char.py`: round-trip, `vocab_size`, unknown char raises `KeyError`, field reassignment raises `FrozenInstanceError`
- [x] 1.3 Implement `CharTokenizer` in `src/demoodle/tokenizers/char.py` — frozen dataclass, `char_to_id: dict[str, int]`, `vocab_size` property, `encode`, `decode`
- [x] 1.4 Confirm tests pass

## 2. train_tokenizer stage

- [x] 2.1 Write failing test: `make_train_tokenizer_stage` run via the runner produces a `CharTokenizer` with vocab covering all corpus characters
- [x] 2.2 Implement `make_train_tokenizer_stage(config_hash: str = "") -> Stage` in `src/demoodle/tokenizers/char.py`
- [x] 2.3 Confirm tests pass

## 3. Wire CharTokenizer into the Artifact union

- [x] 3.1 Import `CharTokenizer` in `src/demoodle/core/types.py` and update the `Artifact` union to `Corpus | CharTokenizer | Dataset | Policy | Metrics`

## 4. Extend artifact cache

- [x] 4.1 Write failing test: `CharTokenizer` round-trips through `save` / `load`
- [x] 4.2 Add `CharTokenizer` import and match arm to `_hash_artifact` in `src/demoodle/shell/persistence.py`
- [x] 4.3 Confirm test passes

## 5. Verification

- [x] 5.1 `uv run ruff format src/ tests/`
- [x] 5.2 `uv run ruff check src/ tests/`
- [x] 5.3 `uv run ty check src/ tests/`
- [x] 5.4 `uv run pytest`

## 6. Documentation

- [x] 6.1 Mark W8 done (✅) in `PLANS.md`
- [x] 6.2 Read `README.md` and update if necessary
- [x] 6.3 Review `CONTEXT.md` and add/update terms if necessary (tokenizer, vocabulary, char-tokenizer)
- [x] 6.4 Read files inside `agents/` and update if necessary
