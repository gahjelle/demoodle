# Spec: Char Tokenizer

## Purpose

Provides a character-level tokenizer (`CharTokenizer`) that maps individual characters to integer IDs and back. The tokenizer is built from the sorted unique characters of a training corpus and satisfies `TokenizerProtocol` structurally, enabling it to be used as a drop-in component in the pipeline.

## Requirements

### Requirement: CharTokenizer encodes and decodes text
The system SHALL provide a `CharTokenizer` frozen dataclass in
`demoodle.tokenizers.char` with a `char_to_id: dict[str, int]` field,
`vocab_size: int` property (equal to `len(char_to_id)`), `encode(text) -> list[int]`,
and `decode(ids) -> list[int] -> str` methods satisfying `TokenizerProtocol`
structurally. The mapping SHALL be built from sorted unique characters in the
training corpus; `\n` is included as a regular vocabulary entry.

#### Scenario: Round-trip preserves text
- **WHEN** `decode(encode(s))` is called for any string drawn from the training alphabet
- **THEN** the result equals `s`

#### Scenario: vocab_size matches mapping length
- **WHEN** a `CharTokenizer` is constructed with a given `char_to_id`
- **THEN** `vocab_size` equals `len(char_to_id)`

#### Scenario: Unknown character raises KeyError
- **WHEN** `encode` is called with a character not in `char_to_id`
- **THEN** a `KeyError` is raised

#### Scenario: CharTokenizer is immutable at the field level
- **WHEN** code attempts to reassign `char_to_id` on a `CharTokenizer` instance
- **THEN** a `FrozenInstanceError` is raised

### Requirement: make_train_tokenizer_stage produces a runnable Stage
The system SHALL provide `make_train_tokenizer_stage(config_hash: str = "") -> Stage`
in `demoodle.tokenizers.char`. The returned `Stage` SHALL have `name="train_tokenizer"`,
`needs=["corpus"]`, `produces=["tokenizer"]`, and a `run` function that builds a
`CharTokenizer` from the sorted unique characters of the input `Corpus.text`.

#### Scenario: Stage produces a CharTokenizer artifact
- **WHEN** the stage is executed via the runner with a `Corpus` input
- **THEN** the output artifacts contain a `CharTokenizer` under the key `"tokenizer"`

#### Scenario: Vocabulary covers all corpus characters
- **WHEN** the stage runs on a corpus with known character set
- **THEN** every character in the corpus is present in `char_to_id`
