# Config

## Purpose

Defines how the application configuration is loaded, validated, and exposed to the rest of the codebase. Configuration is sourced from a committed default TOML file (`demoodle.toml`) and can be partially overridden via environment variables.

## Requirements

### Requirement: Config is importable as a typed singleton
The system SHALL expose a fully-typed `DemoodleConfig` instance at `demoodle.config.config` that is loaded once at startup from the committed default TOML file.

#### Scenario: Import succeeds with defaults
- **WHEN** a module imports `from demoodle.config import config`
- **THEN** `config` is a `DemoodleConfig` instance with all fields populated from `demoodle.toml` defaults

#### Scenario: Config is immutable after load
- **WHEN** code attempts to assign a new value to any field on `config`
- **THEN** a `ValidationError` or `TypeError` is raised (pydantic frozen model)

### Requirement: Config is validated at load time
The system SHALL raise a clear error at startup if `demoodle.toml` contains values that do not match the schema (wrong type, missing required field).

#### Scenario: Invalid field type in TOML
- **WHEN** `demoodle.toml` contains a non-integer value for an integer field (e.g. `n_layers = "two"`)
- **THEN** startup fails with a pydantic `ValidationError` identifying the offending field

### Requirement: Active architecture is selectable via environment variable
The system SHALL read the `DEMOODLE_ARCHITECTURE` environment variable and use it to override `config.architecture.active` at load time.

#### Scenario: Env var overrides TOML value
- **WHEN** `DEMOODLE_ARCHITECTURE=transformer` is set in the environment
- **THEN** `config.architecture.active` equals `"transformer"` regardless of the TOML file value

#### Scenario: Env var absent falls back to TOML
- **WHEN** `DEMOODLE_ARCHITECTURE` is not set
- **THEN** `config.architecture.active` equals the value in `demoodle.toml`

### Requirement: Per-architecture config sections are independently typed
The system SHALL provide separate pydantic models for each architecture's hyperparameters, accessible as sub-fields of `config.architecture`.

#### Scenario: MLP config is accessible and typed
- **WHEN** code accesses `config.architecture.mlp`
- **THEN** the result is an `MLPConfig` instance with fields `embedding_dim`, `context_length`, and `hidden_size`

#### Scenario: Bigram config exists with no architecture-specific hyperparameters
- **WHEN** code accesses `config.architecture.bigram`
- **THEN** the result is a `BigramConfig` instance (may be empty — bigram derives vocab size from the tokenizer artifact, not config)

### Requirement: Shell routing uses active architecture key
The system SHALL select the correct architecture sub-config by reading `config.architecture.active` and looking up the matching field.

#### Scenario: Active key resolves to sub-config
- **WHEN** `config.architecture.active` is `"mlp"`
- **THEN** `getattr(config.architecture, config.architecture.active)` returns the `MLPConfig` instance

### Requirement: Config exposes a corpus section with active selection and per-corpus metadata
The system SHALL provide a `CorpusConfig` section at `config.corpus` with an `active` key and per-corpus entries (`names`, `shakespeare`, `code`), each carrying `url`, `description`, and `license` metadata fields.

#### Scenario: Corpus config is accessible and typed
- **WHEN** code accesses `config.corpus`
- **THEN** the result is a `CorpusConfig` instance with fields `active`, `names`, `shakespeare`, and `code`

#### Scenario: Per-corpus metadata is accessible
- **WHEN** code accesses `config.corpus.names`
- **THEN** the result is a `CorpusEntryConfig` instance with non-empty `url`, `description`, and `license` fields
