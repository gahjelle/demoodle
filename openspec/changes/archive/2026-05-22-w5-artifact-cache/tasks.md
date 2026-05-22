## 1. Update Stage Protocol (breaking change from W4)

- [x] 1.1 Add required `config_hash: str` field (no default) to `Stage` in `src/demoodle/ports/protocols.py`
- [x] 1.2 Update `tests/ports/test_protocols.py` to supply `config_hash=""` in all Stage constructions

## 2. Implement Artifact Hashing

- [x] 2.1 Write failing test: `hash_artifact(Corpus(...))` returns a stable hex string
- [x] 2.2 Implement `_hash_artifact(artifact: Artifact) -> str` in `src/demoodle/shell/persistence.py` with type dispatch for all current Artifact variants (Corpus, Tokenizer, Dataset, Policy, Metrics)
- [x] 2.3 Write failing tests: same artifact → same hash; mutated artifact → different hash for each variant

## 3. Implement cache_key

- [x] 3.1 Write failing test: same (stage, inputs, rng) → same key
- [x] 3.2 Write failing test: different `rng.seed` → different key
- [x] 3.3 Write failing test: different `config_hash` → different key
- [x] 3.4 Write failing test: different input artifact content → different key
- [x] 3.5 Implement `_git_id() -> str` with `subprocess`, catching `FileNotFoundError` and `CalledProcessError`, falling back to `""`; store result as module-level `_GIT_ID`
- [x] 3.6 Implement `cache_key(stage, inputs, rng) -> str` combining `__version__`, `_GIT_ID`, `stage.config_hash`, sorted input artifact hashes, and `rng.seed`
- [x] 3.7 Write test for git fallback: patch `subprocess.run` to raise `FileNotFoundError`; confirm module-level `_GIT_ID` is `""` and `cache_key` still returns a string

## 4. Implement save and load

- [x] 4.1 Write failing test: save then load a `Corpus` artifact → text matches
- [x] 4.2 Write failing test: save then load a `Dataset` artifact → tensors equal
- [x] 4.3 Write failing test: save then load a `Policy` artifact → state dicts equal
- [x] 4.4 Write failing test: `load` with unknown key → returns `None`
- [x] 4.5 Implement `save(key, artifacts, cache_dir)` using `torch.save`; store at `cache_dir/<key>.pt`
- [x] 4.6 Implement `load(key, cache_dir)` using `torch.load`; return `None` if file absent

## 5. Verification

- [x] 5.1 Run `uv run ruff format src/ tests/` and fix any issues
- [x] 5.2 Run `uv run ruff check src/ tests/` and fix any issues
- [x] 5.3 Run `uv run ty check src/ tests/` and fix any type errors
- [x] 5.4 Run `uv run pytest` and confirm all tests pass

## 6. Documentation

- [x] 6.1 Read `README.md`, `PLANS.md`, and files inside `agents/` and update whatever is necessary
- [x] 6.2 Mark W5 as ✅ done in `PLANS.md`
