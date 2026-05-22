## ADDED Requirements

### Requirement: Corpus can be loaded by name from bundled package data
The system SHALL provide a `load_corpus(name: str) -> Corpus` function in `demoodle.data.loaders` that reads `{name}.txt` from the `demoodle.data` package using `importlib.resources` and returns a `Corpus` artifact.

#### Scenario: Load names corpus
- **WHEN** `load_corpus("names")` is called
- **THEN** a `Corpus` instance is returned with non-empty `text` containing approximately 32,000 newline-separated names

#### Scenario: Unknown corpus raises error
- **WHEN** `load_corpus("nonexistent")` is called
- **THEN** a `FileNotFoundError` is raised (propagated from `importlib.resources`)

### Requirement: Active corpus is config-driven
The system SHALL select the corpus to load using `config.corpus.active` as the name argument to `load_corpus`.

#### Scenario: Default active corpus is names
- **WHEN** `config.corpus.active` is `"names"`
- **THEN** `load_corpus(config.corpus.active)` returns a `Corpus` with the names dataset

### Requirement: Corpus text is clean on load
The system SHALL strip trailing whitespace from each line and exclude empty lines when building the `Corpus.text`.

#### Scenario: Trailing newline stripped
- **WHEN** `names.txt` ends with a trailing newline
- **THEN** the resulting `Corpus.text` does not end with `\n`
