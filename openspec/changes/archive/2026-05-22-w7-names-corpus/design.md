## Context

The pipeline runner (W6) accepts initial artifacts as input — the corpus is not produced by a stage, it is the seed. Something upstream (the CLI, W12) must load the corpus and pass it in. W7 provides that loading mechanism. Data must be bundled with the package so the tool works out-of-the-box without a download step, while the config preserves URLs for future optional downloading.

Currently `PathsConfig.data_dir` points to a top-level `data/` directory resolved at runtime via `{project_path}`. This approach does not work when the package is installed from PyPI (no guaranteed project root). `importlib.resources` solves this portably.

## Goals / Non-Goals

**Goals:**
- Bundle `names.txt` inside the Python package (`src/demoodle/data/`)
- Resolve data paths via `importlib.resources` — works installed or from source
- Config-driven corpus selection (`corpus.active`)
- Per-corpus metadata (url, description, license) for documentation and future download support
- Stub `shakespeare` and `code` entries for W14/W15 without bundling their data yet

**Non-Goals:**
- Live downloading from URLs
- Corpus preprocessing or normalization
- Shakespeare or code data files

## Decisions

### Bundle data inside the package (not a top-level `data/` dir)

**Decision:** `src/demoodle/data/names.txt`, resolved via `importlib.resources.files("demoodle.data")`.

**Alternatives considered:**
- Top-level `data/` + `{project_path}` in config — breaks on PyPI installs; runtime path resolution is fragile.
- XDG data dir with a download-on-first-use pattern — adds complexity; overkill for bundled public-domain data.

### Remove `data_dir` from `PathsConfig`

**Decision:** Drop the field entirely now rather than leave dead config.

**Rationale:** Once data is in the package, `data_dir` has no use until a download feature is built. Empty config entries mislead readers. Re-add when download support lands (W14/W15 or later).

### `load_corpus(name: str) -> Corpus`

**Decision:** Single function in `src/demoodle/data/loaders.py` — takes the corpus name string (e.g. `"names"`), reads `{name}.txt` from the package data, returns a `Corpus`.

**Alternatives considered:**
- `load_names()` dedicated function — doesn't scale to W14/W15; would need `load_shakespeare()`, `load_code()` etc.
- Corpus-keyed dict — over-engineered for one function call.

### Per-corpus metadata in config (url, description, license)

**Decision:** `CorpusEntryConfig` has `url: str`, `description: str`, `license: str`. All three present even for stubs (empty strings for `code` where unknown).

**Rationale:** The config doubles as machine-readable documentation. URL supports future download; description and license make provenance visible without hunting for a README.

## Risks / Trade-offs

- **`uv_build` package data inclusion** — Non-`.py` files under `src/` are included by default in hatchling-based builds, but this should be verified. → If `names.txt` is not packaged, add `[tool.hatch.build.targets.wheel] include = ["src/demoodle/data/*.txt"]` to `pyproject.toml`.
- **`names.txt` size (~200 KB)** — Negligible for a dev/demo tool. Not a concern.
- **Stub corpus entries without data** — `load_corpus("shakespeare")` will raise `FileNotFoundError` until W14 bundles the file. This is acceptable and expected; the stubs are config-only.

## Open Questions

None — all decisions resolved during explore session.
