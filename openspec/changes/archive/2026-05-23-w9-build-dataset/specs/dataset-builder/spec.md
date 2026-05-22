## ADDED Requirements

### Requirement: Encode corpus to token ids
The system SHALL provide a `make_build_dataset_stage()` factory that returns a
`Stage` encoding a `Corpus` artifact into a `Dataset` artifact using a
`TokenizerProtocol` implementor. The resulting `Dataset.tokens` SHALL be a 1D
`torch.long` tensor whose length equals the number of characters encoded.

#### Scenario: Produces a Dataset artifact
- **WHEN** the stage is run with a `corpus` and `tokenizer` artifact
- **THEN** it produces a `Dataset` artifact with a 1D `torch.long` tensor

#### Scenario: Correct token sequence
- **WHEN** the corpus text is `"ab\ncd"` and each character maps to a unique id
- **THEN** `dataset.tokens.tolist()` equals `[id_a, id_b, id_newline, id_c, id_d, id_newline]`

#### Scenario: Targets are inputs shifted by one
- **WHEN** the dataset tokens are `t`
- **THEN** `t[1:]` represents the target for each position in `t[:-1]`

### Requirement: Trailing newline normalization
The stage SHALL ensure the corpus text ends with `\n` before encoding, so that
the final name's boundary transition is present in the training sequence.

#### Scenario: Missing trailing newline is added
- **WHEN** `corpus.text` does not end with `\n`
- **THEN** the encoded sequence ends with the token id for `\n`

#### Scenario: Existing trailing newline is not duplicated
- **WHEN** `corpus.text` already ends with `\n`
- **THEN** the encoded sequence contains exactly one trailing `\n` token (not two)

### Requirement: Stage integrates with the runner
The `build_dataset` stage SHALL declare `needs=["corpus", "tokenizer"]` and
`produces=["dataset"]`, and SHALL be cacheable by the existing runner and
persistence layer without any modifications to those components.

#### Scenario: Runner executes the stage
- **WHEN** the runner is given corpus and tokenizer artifacts and the build_dataset stage
- **THEN** it produces a `dataset` artifact in the output dict

#### Scenario: Cache hit on rerun
- **WHEN** the stage is run twice with identical inputs
- **THEN** the second run returns the cached result without re-encoding
