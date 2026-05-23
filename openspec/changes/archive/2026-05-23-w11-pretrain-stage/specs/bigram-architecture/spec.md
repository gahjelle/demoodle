## MODIFIED Requirements

### Requirement: BigramArchitecture satisfies ArchitectureProtocol
The system SHALL provide `BigramArchitecture` in `demoodle.architectures.bigram` with `vocab_size: int` and `context_length: int = 1` bound at construction. It SHALL implement `init_state(rng: RNG) -> Policy`, `forward(policy: Policy, tokens: Seq) -> Output`, and expose `context_length`, satisfying `ArchitectureProtocol` structurally.

#### Scenario: init_state returns a Policy wrapping a BigramModel
- **WHEN** `BigramArchitecture(vocab_size=27).init_state(rng)` is called
- **THEN** it returns a `Policy` whose `model` is a `BigramModel` instance with `weight.shape == (27, 27)`

#### Scenario: init_state is deterministic under fixed seed
- **WHEN** `init_state` is called twice with the same `RNG`
- **THEN** the resulting `Policy` weights are identical

#### Scenario: forward returns logits of correct shape
- **WHEN** `forward(policy, tokens)` is called with any non-empty token sequence
- **THEN** `output.logits.shape == (vocab_size,)` and `output.sampled_ids is None`

#### Scenario: forward uses only the last token
- **WHEN** `forward` is called with sequences of different lengths ending in the same token
- **THEN** the returned logits are identical

#### Scenario: context_length is 1
- **WHEN** `BigramArchitecture` is instantiated with any `vocab_size`
- **THEN** `arch.context_length == 1`
