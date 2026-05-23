## ADDED Requirements

### Requirement: make_pretrain_stage returns a Stage that trains any architecture
The system SHALL provide `make_pretrain_stage(arch, config: PretrainConfig) -> Stage` in `demoodle.training.stages`. The returned `Stage` SHALL have `name="pretrain"`, `needs=["dataset"]`, `produces=["base_policy", "metrics"]`, and a `config_hash` derived from the pretrain config fields plus `arch.vocab_size` and `arch.context_length`.

#### Scenario: Stage produces base_policy and metrics artifacts
- **WHEN** `stage.run({"dataset": dataset}, rng)` is called
- **THEN** the result contains `"base_policy"` (a `Policy`) and `"metrics"` (a `TrainingMetrics`)

#### Scenario: config_hash changes when learning rate changes
- **WHEN** two stages are created with identical args except different `learning_rate`
- **THEN** their `config_hash` fields differ

#### Scenario: config_hash changes when arch.vocab_size changes
- **WHEN** two stages are created with architectures of different `vocab_size`
- **THEN** their `config_hash` fields differ

### Requirement: Training loop minimises cross-entropy loss on the dataset
The pretrain stage SHALL train the architecture's model using Adam and cross-entropy loss over random context windows sampled from the dataset. After `n_steps` steps the mean loss over the final tenth of recorded steps SHALL be lower than over the first tenth.

#### Scenario: Loss decreases over training
- **WHEN** the stage runs for `n_steps` on a dataset
- **THEN** the mean loss over the final tenth of steps is lower than the mean over the first tenth

#### Scenario: Batch sampler uses the provided RNG
- **WHEN** `stage.run` is called twice with the same RNG and dataset
- **THEN** the returned `base_policy` weights are identical

### Requirement: Pretrain stage integrates with the Runner and Artifact Cache
The pretrain stage SHALL round-trip through the Runner and cache correctly: a second run with identical inputs SHALL return the cached `base_policy` and `metrics` without re-executing the training loop.

#### Scenario: Second run is a cache hit
- **WHEN** the Runner executes the pretrain stage twice with the same initial artifacts and RNG
- **THEN** the training loop runs exactly once (the second run is served from cache)

#### Scenario: Changed RNG seed produces a cache miss
- **WHEN** the Runner executes the pretrain stage with two different RNG seeds
- **THEN** the training loop runs twice, producing two distinct `base_policy` artifacts

### Requirement: TrainingMetrics records per-step loss
The `TrainingMetrics` artifact produced by the pretrain stage SHALL contain one loss value per training step, stored as `losses: list[float]`.

#### Scenario: Number of recorded losses equals n_steps
- **WHEN** `make_pretrain_stage` is configured with `n_steps=100` and the stage runs
- **THEN** `metrics.losses` has length 100
