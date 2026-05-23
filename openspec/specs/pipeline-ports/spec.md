# Spec: Pipeline Ports

## Purpose

Defines the protocols and concrete types that form the ports layer of Demoodle's pipeline. These abstractions decouple the core domain from specific implementations of tokenization, model architecture, and inference, enabling interchangeable components and testability.

## Requirements

### Requirement: TokenizerProtocol defines the encode/decode contract
The system SHALL provide a `TokenizerProtocol` in `demoodle.ports.protocols` with methods `encode(text: str) -> list[int]`, `decode(ids: list[int]) -> str`, and an integer attribute `vocab_size`. This is the behavioral contract; the `Tokenizer` artifact in `core.types` remains a separate, minimal data carrier.

#### Scenario: Dummy class satisfies TokenizerProtocol
- **WHEN** a class implements `encode`, `decode`, and `vocab_size`
- **THEN** it type-checks as a valid `TokenizerProtocol` with no errors

#### Scenario: Round-trip property holds for conforming implementations
- **WHEN** a `TokenizerProtocol` implementation encodes then decodes a string
- **THEN** the result equals the original string

### Requirement: ArchitectureProtocol defines the training interface
The system SHALL provide an `ArchitectureProtocol` with `init_state(rng: RNG) -> Policy` and `forward(policy: Policy, tokens: Seq) -> Output`. Architecture classes bind their hyperparameters (vocab size, hidden dimensions, etc.) at construction time; `init_state` is a pure function of `rng` only.

#### Scenario: Dummy class satisfies ArchitectureProtocol
- **WHEN** a class implements `init_state` and `forward` with compatible return types
- **THEN** it type-checks as a valid `ArchitectureProtocol` with no errors

### Requirement: InspectableProtocol defines the inference and inspection interface
The system SHALL provide an `InspectableProtocol` where `call(seq: Seq, policy: Policy, rng: RNG, temperature: float, top_k: int | None = None, top_p: float | None = None) -> Output` is required and `explain(seq: Seq, policy: Policy) -> dict[str, Any]` is optional with a default body returning `{}`. Implementations that do not override `explain` inherit the default.

Both `call` and `explain` receive `policy` explicitly. Architectures are stateless config/logic; all model state lives in `Policy`. This mirrors `ArchitectureProtocol.forward(policy, tokens)` — no architecture holds a `Policy` reference internally.

`call` returns an `Output` containing both `logits` (the full distribution over the vocabulary) and `sampled_ids` (the single drawn token id), enabling front ends to visualise the probability distribution without a second forward pass.

`top_k` and `top_p` are optional sampling controls. Implementations MAY ignore them when not applicable (e.g. at small vocabulary sizes where they have no meaningful effect), but SHALL accept them without error.

#### Scenario: Dummy class with only `call` satisfies InspectableProtocol
- **WHEN** a class implements only `call(seq, policy, rng, temperature, top_k=None, top_p=None)`
- **THEN** it type-checks as a valid `InspectableProtocol` and `explain(seq, policy)` returns `{}`

#### Scenario: Overriding `explain` returns custom data
- **WHEN** an implementation overrides `explain(seq, policy)` to return attention weights
- **THEN** calling `explain(seq, policy)` returns that implementation's data, not `{}`

#### Scenario: call receives policy explicitly
- **WHEN** a conforming implementation's `call` is invoked
- **THEN** it receives the model state via `policy`, not from any internal field on the architecture

### Requirement: Stage is a frozen dataclass with a typed run callable
The system SHALL provide a `Stage` frozen dataclass with fields `name: str`, `needs: list[str]`, `produces: list[str]`, `config_hash: str`, and `run: Callable[[dict[str, Artifact], RNG], dict[str, Artifact]]`. `Stage` is not a Protocol; it is a concrete, immutable value.

The `config_hash` field is required with no default. Stage authors SHALL compute it by serialising the pydantic sub-configs that affect the stage's output (using `model_dump_json()`) and hashing the result. Stages whose output is independent of all config SHALL pass `config_hash=""` explicitly.

#### Scenario: Stage cannot be mutated after construction
- **WHEN** a `Stage` instance is constructed
- **THEN** attempting to reassign any field raises `FrozenInstanceError`

#### Scenario: Stage.run is a pure function of its inputs
- **WHEN** `stage.run` is called with the same artifact dict and the same `RNG`
- **THEN** it returns the same output artifacts

#### Scenario: Stage construction requires config_hash
- **WHEN** a `Stage` is constructed without supplying `config_hash`
- **THEN** a `TypeError` is raised

#### Scenario: Stage with no config sensitivity uses empty string
- **WHEN** a stage's output does not depend on any config value
- **THEN** it is constructed with `config_hash=""`

### Requirement: All protocols and Stage are importable from demoodle.ports
The system SHALL export `TokenizerProtocol`, `ArchitectureProtocol`, `InspectableProtocol`, and `Stage` from `demoodle.ports`.

#### Scenario: Import from package root
- **WHEN** a module executes `from demoodle.ports import TokenizerProtocol, ArchitectureProtocol, InspectableProtocol, Stage`
- **THEN** all four names resolve without error
