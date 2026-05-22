## Why

The pipeline runner (W6) needs to skip re-running stages whose inputs haven't changed. Without a cache, every run re-trains from scratch — making live demos impractical and development slow. W5 provides the persistence layer that W6 will depend on.

## What Changes

- **NEW** `shell/persistence.py` — content-addressed artifact store with `cache_key`, `save`, and `load` functions
- **NEW** `tests/shell/test_persistence.py` — round-trip and key-sensitivity tests
- **BREAKING** `Stage` frozen dataclass gains a non-optional `config_hash: str` field — stage authors must declare which config values affect their outputs
- Updated `tests/ports/test_protocols.py` to supply `config_hash` in all Stage constructions

## Capabilities

### New Capabilities

- `artifact-cache`: Content-addressed save/load of pipeline artifacts to a local cache directory, keyed by stage name, code version, config snapshot, input artifact hashes, and RNG seed.

### Modified Capabilities

- `pipeline-ports`: `Stage` gains a required `config_hash: str` field. This is a breaking change to the existing protocol — all Stage constructions must be updated.

## Impact

- `src/demoodle/ports/protocols.py` — `Stage` dataclass updated
- `src/demoodle/shell/persistence.py` — new module
- `tests/ports/test_protocols.py` — existing test updated for new field
- `tests/shell/test_persistence.py` — new test module
- No new dependencies; uses `torch`, `hashlib`, `subprocess`, `pathlib` (all already available)

## Non-goals

- No cache eviction or size management — cache is append-only; users clear it manually
- No cross-machine or networked cache sharing
- No human-readable cache format — `torch.save` (pickle-based) is sufficient for a local dev cache
- No cache for partial stage outputs or streaming results
