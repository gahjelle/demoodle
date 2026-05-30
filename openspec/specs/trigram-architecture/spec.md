# Spec: Trigram Architecture

## Purpose

TBD — documents the trigram model architecture implementation, including the `TrigramArchitecture` class that satisfies both `ArchitectureProtocol` and `InspectableProtocol`, and the underlying `TrigramModel` nn.Module that performs a two-token context lookup.

## Requirements

### Requirement: TrigramArchitecture satisfies ArchitectureProtocol
The system SHALL provide `TrigramArchitecture` in `demoodle.architectures.trigram` with `vocab_size: int` and `context_length: int = 2` bound at construction. It SHALL implement `init_state(rng: RNG) -> Policy`, `forward(policy: Policy, tokens: Seq) -> Output`, and expose `context_length`, satisfying `ArchitectureProtocol` structurally.

#### Scenario: init_state returns a Policy wrapping a TrigramModel
- **WHEN** `TrigramArchitecture(vocab_size=27).init_state(rng)` is called
- **THEN** it returns a `Policy` whose `model` is a `TrigramModel` instance with `weight.shape == (27, 27, 27)`

#### Scenario: init_state is deterministic under fixed seed
- **WHEN** `init_state` is called twice with the same `RNG`
- **THEN** the resulting `Policy` weights are identical

#### Scenario: forward returns logits of correct shape
- **WHEN** `forward(policy, tokens)` is called with a token sequence of length >= 2
- **THEN** `output.logits.shape == (vocab_size,)` and `output.sampled_ids is None`

#### Scenario: forward uses only the last two tokens
- **WHEN** `forward` is called with sequences of different lengths sharing the same last two tokens
- **THEN** the returned logits are identical

#### Scenario: context_length is 2
- **WHEN** `TrigramArchitecture` is instantiated with any `vocab_size`
- **THEN** `arch.context_length == 2`

### Requirement: TrigramArchitecture satisfies InspectableProtocol
`TrigramArchitecture` SHALL implement `call(seq, policy, rng, temperature, top_k=None, top_p=None) -> Output` and `explain(seq, policy) -> dict`. `call` SHALL return an `Output` with both `logits` and `sampled_ids` set. `explain` SHALL return `{}`.

#### Scenario: call returns sampled_ids
- **WHEN** `call(seq, policy, temperature=1.0)` is called with a 2-token sequence
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

### Requirement: TrigramModel is a thin nn.Module wrapping a single nn.Parameter
The system SHALL provide `TrigramModel(nn.Module)` in `demoodle.architectures.trigram` with a single `weight: nn.Parameter` of shape `(vocab_size, vocab_size, vocab_size)`. Its `forward(x: torch.Tensor) -> torch.Tensor` SHALL accept either a 1-D tensor of shape `(2,)` (single prediction) or a 2-D tensor of shape `(batch, 2)` (batched training), returning logits of shape `(vocab_size,)` or `(batch, vocab_size)` respectively.

#### Scenario: TrigramModel forward performs a two-token lookup (single)
- **WHEN** `TrigramModel(vocab_size=5).forward(torch.tensor([1, 3]))` is called
- **THEN** the output equals `model.weight[1, 3]` and has shape `(5,)`

#### Scenario: TrigramModel forward handles batched input
- **WHEN** `TrigramModel(vocab_size=5).forward(torch.tensor([[1, 3], [0, 2]]))` is called
- **THEN** the output has shape `(2, 5)` and row 0 equals `model.weight[1, 3]`

#### Scenario: TrigramModel has exactly one parameter group
- **WHEN** `list(model.parameters())` is called on a `TrigramModel`
- **THEN** there is exactly one parameter with shape `(vocab_size, vocab_size, vocab_size)`

### Requirement: Trigram is selectable via config
The system SHALL accept `architecture.active = "trigram"` in `demoodle.toml` and instantiate `TrigramArchitecture` accordingly. An empty `[architecture.trigram]` section SHALL be present in the default config.

#### Scenario: _make_arch creates TrigramArchitecture when active is "trigram"
- **WHEN** `config.architecture.active == "trigram"` and `_make_arch(vocab_size)` is called
- **THEN** a `TrigramArchitecture` instance is returned with the correct `vocab_size`
