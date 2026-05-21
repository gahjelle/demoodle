## ADDED Requirements

### Requirement: Seq type alias
The module SHALL export `Seq` as a type alias for `torch.Tensor`, representing a 1D integer tensor of token IDs.

#### Scenario: Seq is a tensor alias
- **WHEN** code imports `Seq` from `demoodle.core.types`
- **THEN** `Seq` is assignable to and from `torch.Tensor`

### Requirement: Output frozen dataclass
The module SHALL export `Output` as a frozen dataclass with fields `logits: torch.Tensor` and `sampled_ids: torch.Tensor | None` (defaulting to `None`).

#### Scenario: Output is constructable with logits only
- **WHEN** `Output(logits=some_tensor)` is called
- **THEN** a valid `Output` is returned with `sampled_ids` equal to `None`

#### Scenario: Output is immutable
- **WHEN** code attempts to assign to any field of an `Output` instance
- **THEN** a `FrozenInstanceError` is raised

### Requirement: Corpus frozen dataclass
The module SHALL export `Corpus` as a frozen dataclass with field `text: str` holding the raw, unsegmented text of a corpus.

#### Scenario: Corpus is constructable
- **WHEN** `Corpus(text="alice\nbob\n")` is called
- **THEN** a valid `Corpus` is returned with the given text

#### Scenario: Corpus is immutable
- **WHEN** code attempts to assign to `text` on a `Corpus` instance
- **THEN** a `FrozenInstanceError` is raised

### Requirement: Tokenizer frozen dataclass
The module SHALL export `Tokenizer` as a frozen dataclass with field `vocab_size: int`, serving as a minimal base for concrete tokenizer artifacts.

#### Scenario: Tokenizer is constructable
- **WHEN** `Tokenizer(vocab_size=27)` is called
- **THEN** a valid `Tokenizer` is returned

#### Scenario: Tokenizer is immutable
- **WHEN** code attempts to assign to any field of a `Tokenizer` instance
- **THEN** a `FrozenInstanceError` is raised

### Requirement: Dataset frozen dataclass
The module SHALL export `Dataset` as a frozen dataclass with field `tokens: torch.Tensor`, representing the full encoded corpus as a flat sequence of token IDs.

#### Scenario: Dataset is constructable
- **WHEN** `Dataset(tokens=some_1d_tensor)` is called
- **THEN** a valid `Dataset` is returned

#### Scenario: Dataset is immutable
- **WHEN** code attempts to assign to `tokens` on a `Dataset` instance
- **THEN** a `FrozenInstanceError` is raised

### Requirement: Policy frozen dataclass
The module SHALL export `Policy` as a frozen dataclass with fields `model: torch.nn.Module` and `value_head: torch.nn.Module | None` (defaulting to `None`). The `value_head` field is reserved for PPO and SHALL be `None` for all non-PPO policies.

#### Scenario: Policy is constructable with model only
- **WHEN** `Policy(model=some_module)` is called
- **THEN** a valid `Policy` is returned with `value_head` equal to `None`

#### Scenario: Policy is immutable at the field level
- **WHEN** code attempts to reassign `policy.model` or `policy.value_head`
- **THEN** a `FrozenInstanceError` is raised

### Requirement: Metrics frozen dataclass
The module SHALL export `Metrics` as a frozen dataclass with field `losses: list[float]` recording the per-step training loss curve.

#### Scenario: Metrics is constructable
- **WHEN** `Metrics(losses=[1.0, 0.8, 0.6])` is called
- **THEN** a valid `Metrics` is returned

#### Scenario: Metrics is immutable
- **WHEN** code attempts to assign to `losses` on a `Metrics` instance
- **THEN** a `FrozenInstanceError` is raised

### Requirement: Artifact union type
The module SHALL export `Artifact` as a union type `Corpus | Tokenizer | Dataset | Policy | Metrics`. The union SHALL include stub comments indicating that `RewardModel` and `PreferenceData` will be added in Milestone 5.

#### Scenario: Each variant is a valid Artifact
- **WHEN** any of `Corpus`, `Tokenizer`, `Dataset`, `Policy`, or `Metrics` is type-checked against `Artifact`
- **THEN** the type checker accepts it without error
