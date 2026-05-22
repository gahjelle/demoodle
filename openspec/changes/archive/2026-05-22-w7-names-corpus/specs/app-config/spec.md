## ADDED Requirements

### Requirement: Config exposes a corpus section with active selection and per-corpus metadata
The system SHALL provide a `CorpusConfig` section at `config.corpus` with an `active` key and per-corpus entries (`names`, `shakespeare`, `code`), each carrying `url`, `description`, and `license` metadata fields.

#### Scenario: Corpus config is accessible and typed
- **WHEN** code accesses `config.corpus`
- **THEN** the result is a `CorpusConfig` instance with fields `active`, `names`, `shakespeare`, and `code`

#### Scenario: Per-corpus metadata is accessible
- **WHEN** code accesses `config.corpus.names`
- **THEN** the result is a `CorpusEntryConfig` instance with non-empty `url`, `description`, and `license` fields

## REMOVED Requirements

### Requirement: Paths config includes data_dir
**Reason:** Data is now bundled inside the package and resolved via `importlib.resources`. A filesystem path is no longer needed and would mislead readers.
**Migration:** Use `importlib.resources.files("demoodle.data")` to locate bundled data files. Future download support will introduce its own path config when needed.
