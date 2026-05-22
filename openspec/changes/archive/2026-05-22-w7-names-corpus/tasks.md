## 1. Config schema and TOML

- [x] 1.1 Add `CorpusEntryConfig(url, description, license)` and `CorpusConfig(active, names, shakespeare, code)` to `src/demoodle/config/schemas.py`
- [x] 1.2 Remove `data_dir` from `PathsConfig` in `src/demoodle/config/schemas.py`
- [x] 1.3 Add `[corpus]` section to `src/demoodle/config/demoodle.toml` with `active = "names"` and per-corpus entries (url, description, license for names and shakespeare; empty strings for code)
- [x] 1.4 Remove `data_dir` from `[paths]` in `src/demoodle/config/demoodle.toml`

## 2. Data package and names corpus

- [x] 2.1 Create `src/demoodle/data/__init__.py`
- [x] 2.2 Obtain and commit `names.txt` (~32K SSA public-domain names, one per line) to `src/demoodle/data/names.txt`

## 3. Loader implementation

- [x] 3.1 Write failing test: `load_corpus("names")` returns non-empty `Corpus`, line count ~32K
- [x] 3.2 Implement `load_corpus(name: str) -> Corpus` in `src/demoodle/data/loaders.py` using `importlib.resources.files("demoodle.data")`
- [x] 3.3 Confirm test passes

## 4. Verification

- [x] 4.1 `uv run ruff format src/ tests/`
- [x] 4.2 `uv run ruff check src/ tests/`
- [x] 4.3 `uv run ty check src/ tests/`
- [x] 4.4 `uv run pytest`

## 5. Documentation

- [x] 5.1 Mark W7 done (✅) in `PLANS.md`
- [x] 5.2 Review `README.md` and update if necessary
