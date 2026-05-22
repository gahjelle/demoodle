## Why

Pipeline stages that do random work (model init, data shuffling, sampling) currently have no principled way to manage randomness — they would rely on PyTorch's global seed state, making runs fragile and hard to reproduce. An explicit `RNG` value type threads randomness through the pipeline the same way artifacts do: explicitly, immutably, and reproducibly.

## What Changes

- Add `core/rng.py` with a frozen `RNG` dataclass
- `RNG.split()` produces two independent child RNGs deterministically (via hashlib)
- `RNG.generator()` returns a seeded `torch.Generator` for use in torch ops
- No global seeding anywhere in the codebase

## Capabilities

### New Capabilities

- `rng`: Deterministic, explicit RNG value type for threading randomness through pipeline stages

### Modified Capabilities

<!-- No existing spec-level requirements are changing -->

## Impact

- New file: `src/demoodle/core/rng.py`
- New tests: `tests/core/test_rng.py`
- No changes to existing types, config, or any stage code
- Downstream stages (W5 persistence, W6 runner, W10 bigram) will accept `RNG` as a parameter — that wiring happens in those work items

## Non-goals

- Global seed management or convenience wrappers around `torch.manual_seed`
- Numpy or stdlib `random` integration
- RNG serialization or inclusion in the `Artifact` union (the seed value, if needed for reproducibility logging, belongs in `Metrics`)
