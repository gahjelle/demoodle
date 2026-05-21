## 1. Dependency

- [x] 1.1 Run `uv add pydantic` to add pydantic v2 as a project dependency

## 2. Schema

- [x] 2.1 Create `src/demoodle/config/schemas.py` with a `StrictModel(BaseModel)` base class using `ConfigDict(frozen=True, extras="forbid")`, then define `BigramConfig`, `MLPConfig`, `TransformerConfig` inheriting from it
- [x] 2.2 Add `CharConfig` and `BPEConfig` tokenizer sub-models inheriting from `StrictModel`
- [x] 2.3 Add `ArchitecturesConfig` model (inheriting `StrictModel`) with `active: str` and one field per architecture sub-model
- [x] 2.4 Add `TokenizersConfig`, `TrainingConfig`, and `PathsConfig` models inheriting from `StrictModel`
- [x] 2.5 Add top-level `DemoodleConfig` model inheriting from `StrictModel`, composing all section models

## 3. Default configuration

- [x] 3.1 Create `src/demoodle/config/demoodle.toml` with `[architecture]`, `[architecture.bigram]`, `[architecture.mlp]`, `[architecture.transformer]` sections and sensible defaults
- [x] 3.2 Add `[tokenizer]`, `[tokenizer.char]`, `[tokenizer.bpe]` sections to the TOML
- [x] 3.3 Add `[training]` and `[paths]` sections to the TOML

## 4. Config loader

- [x] 4.1 Create `src/demoodle/config/__init__.py` that loads `demoodle.toml` via `configaroo`, applies `DEMOODLE_ARCHITECTURE` env override, and exposes typed `config: DemoodleConfig` singleton

## 5. Tests

- [x] 5.1 Write a test that imports `from demoodle.config import config` and asserts it is a `DemoodleConfig` instance with expected default values
- [x] 5.2 Write a test that setting `DEMOODLE_ARCHITECTURE=mlp` makes `config.architecture.active == "mlp"` (monkeypatch env, reimport or reload)
- [x] 5.3 Write a test that `config.architecture.mlp` is an `MLPConfig` with correct typed fields
- [x] 5.4 Write a test that `getattr(config.architecture, config.architecture.active)` returns the correct sub-config for each active value

## 6. Verification

- [x] 6.1 Run `uv run ruff format src/ tests/` and fix any formatting issues
- [x] 6.2 Run `uv run ruff check src/ tests/` and fix any lint errors
- [x] 6.3 Run `uv run ty check src/ tests/` and fix any type errors
- [x] 6.4 Run `uv run pytest` and confirm all tests pass

## 7. Documentation

- [x] 7.1 Read `README.md`, `PLANS.md`, and files in `agents/` — update anything affected by the decision to drop the core `Config` dataclass in favour of pydantic section models
- [x] 7.2 In `PLANS.md`, add ✅ to the W2 title and note that `Config` is now `DemoodleConfig` from `demoodle.config`
