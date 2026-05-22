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
The system SHALL provide an `ArchitectureProtocol` with `init_state` (returns a `Policy`) and `forward(policy: Policy, tokens: Seq) -> Output`. The exact signature of `init_state` is left open for W10 to specify; the protocol declares its presence.

#### Scenario: Dummy class satisfies ArchitectureProtocol
- **WHEN** a class implements `init_state` and `forward` with compatible return types
- **THEN** it type-checks as a valid `ArchitectureProtocol` with no errors

### Requirement: InspectableProtocol defines the inference and inspection interface
The system SHALL provide an `InspectableProtocol` where `call(seq: Seq, temperature: float) -> int` is required and `explain() -> dict[str, Any]` is optional with a default body returning `{}`. Implementations that do not override `explain` inherit the default.

#### Scenario: Dummy class with only `call` satisfies InspectableProtocol
- **WHEN** a class implements only `call`
- **THEN** it type-checks as a valid `InspectableProtocol` and `explain()` returns `{}`

#### Scenario: Overriding `explain` returns custom data
- **WHEN** an implementation overrides `explain` to return attention weights
- **THEN** calling `explain()` returns that implementation's data, not `{}`

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
