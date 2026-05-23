## MODIFIED Requirements

### Requirement: Config exposes per-stage training hyperparameters under nested sections
The system SHALL restructure the `[training]` config section so that hyperparameters are nested under named sub-sections, one per training stage. The `TrainingConfig` pydantic model SHALL contain a `pretrain: PretrainConfig` field. `PretrainConfig` SHALL have fields `learning_rate: float`, `batch_size: int`, and `n_steps: int`.

There is no `active` key on `TrainingConfig` — training stages are sequential rather than mutually exclusive.

#### Scenario: Pretrain config is accessible and typed
- **WHEN** code accesses `config.training.pretrain`
- **THEN** the result is a `PretrainConfig` instance with `learning_rate`, `batch_size`, and `n_steps` fields

#### Scenario: Config rejects unrecognised training fields
- **WHEN** `demoodle.toml` contains an unrecognised field under `[training.pretrain]`
- **THEN** startup fails with a pydantic `ValidationError` identifying the offending field

#### Scenario: Import succeeds with defaults
- **WHEN** a module imports `from demoodle.config import config`
- **THEN** `config.training.pretrain.learning_rate` is a positive float
