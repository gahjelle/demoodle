## MODIFIED Requirements

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
