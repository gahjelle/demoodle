## Why

The day-one slice needs a real training loop: without it the bigram architecture exists but cannot learn anything, and the CLI (W12) has no trained policy to load. The pretrain stage is also the foundation every post-training milestone (SFT, DPO, PPO) builds on, so its design must be generic enough to train future architectures without modification.

## What Changes

- Add `context_length: int` to `ArchitectureProtocol` and `BigramArchitecture` (`context_length = 1`), making the training loop architecture-agnostic from the start.
- Restructure `[training]` config from a flat section to nested `[training.pretrain]`, with a dedicated `PretrainConfig` pydantic model — consistent with how `[architecture]` and `[tokenizer]` handle per-variant params.
- Add `training/stages.py` with `make_pretrain_stage(arch, config: PretrainConfig) -> Stage` — a generic Adam + cross-entropy loop that closes over the architecture and config, produces `base_policy` and `metrics`, and caches by hashing all inputs that affect the result.

## Capabilities

### New Capabilities

- `pretrain-stage`: A pipeline stage that trains any architecture on a dataset, returning a `Policy` and `Metrics`. Generic over architecture via `context_length`; caches correctly; loss decreases on names.

### Modified Capabilities

- `pipeline-ports`: `ArchitectureProtocol` gains a required `context_length: int` field.
- `bigram-architecture`: `BigramArchitecture` gains `context_length: int = 1`.
- `app-config`: `[training]` is restructured to `[training.pretrain]`; schema gains `PretrainConfig` and updated `TrainingConfig`.

## Non-goals

- Training MLP or transformer (W13, W16) — `context_length` is added now so those stages need no edits, but the models themselves are not built here.
- SFT, DPO, or PPO training loops (W22, W26, W27).
- A CLI to invoke training (W12).
- Hyperparameter search or scheduler support.

## Impact

- `src/demoodle/ports/protocols.py` — `ArchitectureProtocol` extended
- `src/demoodle/architectures/bigram.py` — `BigramArchitecture` extended
- `src/demoodle/config/schemas.py` — new `PretrainConfig`, updated `TrainingConfig`
- `src/demoodle/config/demoodle.toml` — `[training]` → `[training.pretrain]`
- `src/demoodle/training/stages.py` — new file
- `tests/training/test_stages.py` — new test file
- `tests/architectures/test_bigram.py` — updated for `context_length`
- `tests/ports/test_protocols.py` — updated for `context_length`
