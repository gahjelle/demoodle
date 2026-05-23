## 1. Rename Metrics to TrainingMetrics

- [x] 1.1 Rename `Metrics` to `TrainingMetrics` in `src/demoodle/core/types.py` and update the `Artifact` union
- [x] 1.2 Update `src/demoodle/shell/persistence.py` — rename the `Metrics` branch in `_hash_artifact` and all imports
- [x] 1.3 Update all other references (`tests/`, any other imports) to use `TrainingMetrics`

## 2. Restructure training config

- [x] 2.1 Add `PretrainConfig(learning_rate: float, batch_size: int, n_steps: int)` to `src/demoodle/config/schemas.py`
- [x] 2.2 Replace the flat `TrainingConfig` with `TrainingConfig(pretrain: PretrainConfig)` in `schemas.py`
- [x] 2.3 Replace `[training]` with `[training.pretrain]` in `src/demoodle/config/demoodle.toml`
- [x] 2.4 Update `src/demoodle/shell/persistence.py` `config_hash` computation if it references `config.training` directly

## 3. Add context_length to ArchitectureProtocol and BigramArchitecture

- [x] 3.1 Add `context_length: int` to `ArchitectureProtocol` in `src/demoodle/ports/protocols.py`
- [x] 3.2 Add `context_length: int = 1` to `BigramArchitecture` in `src/demoodle/architectures/bigram.py`
- [x] 3.3 Add a test that `BigramArchitecture.context_length == 1` in `tests/architectures/test_bigram.py`
- [x] 3.4 Update the dummy-class protocol test in `tests/ports/test_protocols.py` to include `context_length`

## 4. Implement make_pretrain_stage

- [x] 4.1 Create `src/demoodle/training/__init__.py` and `src/demoodle/training/stages.py`
- [x] 4.2 Implement `make_pretrain_stage(arch, config: PretrainConfig) -> Stage` with correct `name`, `needs`, `produces`, and `config_hash`
- [x] 4.3 Implement the training loop: random context-window batching via `arch.context_length`, Adam optimizer, cross-entropy loss, step-level loss recording
- [x] 4.4 Return `{"base_policy": Policy, "metrics": TrainingMetrics}` from `stage.run`

## 5. Test pretrain stage

- [x] 5.1 Create `tests/training/__init__.py` and `tests/training/test_stages.py`
- [x] 5.2 Test that `stage.run` returns `base_policy` (Policy) and `metrics` (TrainingMetrics) with correct types
- [x] 5.3 Test that loss decreases: mean of last tenth of `metrics.losses` < mean of first tenth
- [x] 5.4 Test that `stage.run` is deterministic under a fixed RNG
- [x] 5.5 Test that `config_hash` changes when `learning_rate` or `vocab_size` changes
- [x] 5.6 Test cache integration: run via `runner.run` twice; verify training loop executes once

## 6. Verify and clean up

- [x] 6.1 `uv run ruff format src/ tests/`
- [x] 6.2 `uv run ruff check src/ tests/`
- [x] 6.3 `uv run ty check src/ tests/`
- [x] 6.4 `uv run pytest`
- [x] 6.5 Mark W11 as ✅ in `PLANS.md`
- [x] 6.6 Review `CONTEXT.md` — update `ArchitectureProtocol` entry to include `context_length`; confirm `TrainingMetrics` entry matches the renamed type
