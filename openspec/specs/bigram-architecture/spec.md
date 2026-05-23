# Spec: Bigram Architecture

## Purpose

TBD — documents the bigram model architecture implementation, including the `BigramArchitecture` class that satisfies both `ArchitectureProtocol` and `InspectableProtocol`, the underlying `BigramModel` nn.Module, and the sampling helper function.

## Requirements

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

### Requirement: BigramArchitecture satisfies InspectableProtocol
`BigramArchitecture` SHALL implement `call(seq, policy, rng, temperature, top_k=None, top_p=None) -> Output` and `explain(seq, policy) -> dict`. `call` SHALL return an `Output` with both `logits` and `sampled_ids` set. `explain` SHALL return `{}`.

#### Scenario: call returns sampled_ids
- **WHEN** `call(seq, policy, temperature=1.0)` is called
- **THEN** `output.sampled_ids` is a scalar tensor with a valid token id in `[0, vocab_size)`

#### Scenario: temperature affects distribution sharpness
- **WHEN** `call` is called with `temperature=0.01` vs `temperature=10.0` over many draws
- **THEN** low temperature produces less variety (more concentrated on the argmax token)

#### Scenario: top_k restricts the sampled set
- **WHEN** `call` is called with `top_k=1`
- **THEN** `sampled_ids` always equals the argmax of the logits

#### Scenario: top_p restricts the sampled set
- **WHEN** `call` is called with `top_p=0.0`
- **THEN** `sampled_ids` always equals the argmax of the logits

#### Scenario: explain returns empty dict
- **WHEN** `explain(seq, policy)` is called on any input
- **THEN** it returns `{}`

### Requirement: BigramModel is a thin nn.Module wrapping a single nn.Parameter
The system SHALL provide `BigramModel(nn.Module)` in `demoodle.architectures.bigram` with a single `weight: nn.Parameter` of shape `(vocab_size, vocab_size)`. Its `forward(x: torch.Tensor) -> torch.Tensor` SHALL return `self.weight[x]` — a row lookup.

#### Scenario: BigramModel forward performs a row lookup
- **WHEN** `BigramModel(vocab_size=5).forward(torch.tensor(2))` is called
- **THEN** the output equals `model.weight[2]` and has shape `(5,)`

#### Scenario: BigramModel has exactly one parameter group
- **WHEN** `list(model.parameters())` is called on a `BigramModel`
- **THEN** there is exactly one parameter with shape `(vocab_size, vocab_size)`

### Requirement: Sampling helper is available as a module-level pure function
The system SHALL provide `_sample(logits, temperature, top_k, top_p, generator) -> torch.Tensor` in `demoodle.architectures.bigram`. It SHALL be a pure function: given the same inputs and generator state, it returns the same token id.

#### Scenario: _sample with temperature=1 and no filters samples from full distribution
- **WHEN** `_sample` is called with `top_k=None, top_p=None, temperature=1.0`
- **THEN** any token id in `[0, vocab_size)` can be returned (no hard exclusions)

#### Scenario: _sample with top_k=1 always returns argmax
- **WHEN** `_sample` is called with `top_k=1`
- **THEN** the returned tensor equals `logits.argmax()`
