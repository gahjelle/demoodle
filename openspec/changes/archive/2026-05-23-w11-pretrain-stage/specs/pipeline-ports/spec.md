## MODIFIED Requirements

### Requirement: ArchitectureProtocol defines the training interface
The system SHALL provide an `ArchitectureProtocol` with `vocab_size: int`, `context_length: int`, `init_state(rng: RNG) -> Policy`, and `forward(policy: Policy, tokens: Seq) -> Output`. Architecture classes bind their hyperparameters at construction time; `init_state` is a pure function of `rng` only.

`vocab_size` is the number of tokens in the vocabulary; it determines the output dimension of the model and is included in the pretrain stage's `config_hash` to invalidate the cache when the vocabulary changes.

`context_length` is the number of preceding tokens used as input for a single prediction. It governs how wide a context window the training loop extracts from the `Dataset`: `input = tokens[i : i + context_length]`, `target = tokens[i + context_length]`. For the bigram, `context_length = 1`.

#### Scenario: Dummy class satisfies ArchitectureProtocol
- **WHEN** a class implements `vocab_size`, `context_length`, `init_state`, and `forward` with compatible types
- **THEN** it type-checks as a valid `ArchitectureProtocol` with no errors

#### Scenario: vocab_size is a positive integer
- **WHEN** any conforming architecture is instantiated
- **THEN** `arch.vocab_size >= 1`

#### Scenario: context_length is a positive integer
- **WHEN** any conforming architecture is instantiated
- **THEN** `arch.context_length >= 1`
