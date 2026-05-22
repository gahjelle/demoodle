## Why

The pipeline needs a real corpus to train on. Without committed data, nothing downstream (tokenizer, dataset, model) can be built or tested. Names is the canonical first corpus — small, public-domain, and produces visible results (generated names) that make the day-one demo tangible.

## What Changes

- Add `src/demoodle/data/` subpackage with `names.txt` bundled (~32K SSA names, one per line)
- Add `load_corpus(name: str) -> Corpus` loader using `importlib.resources`
- Add `[corpus]` section to config with `active`, per-corpus `url`/`description`/`license`
- Stub `shakespeare` and `code` corpus entries (urls provided, data not yet bundled)
- **BREAKING**: Remove `data_dir` from `PathsConfig` — data now resolved via `importlib.resources`, not filesystem path

## Capabilities

### New Capabilities

- `corpus-loader`: Load a named corpus from bundled package data into a `Corpus` artifact; config-driven selection of active corpus; per-corpus metadata (url, description, license)

### Modified Capabilities

- `app-config`: Add `CorpusConfig` section; remove `data_dir` from `PathsConfig`

## Impact

- New: `src/demoodle/data/__init__.py`, `src/demoodle/data/names.txt`, `src/demoodle/data/loaders.py`
- Modified: `src/demoodle/config/schemas.py`, `src/demoodle/config/demoodle.toml`
- Downstream W8+ stages receive `Corpus` from `load_corpus(config.corpus.active)` — no runner or stage edits needed

## Non-goals

- Downloading corpora from URLs (the `url` field is metadata for future use, not a live fetch)
- Shakespeare or code corpus data files (stubs only — W14, W15)
- Corpus preprocessing or normalization beyond stripping trailing whitespace
