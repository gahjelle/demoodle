## Why

Trainadillo's T1–T3 foundation (Tensor, creation functions, RNG) is complete. T4 adds the random tensor factories that consuming code (`_sample()` in bigram.py and future training utilities) need — specifically `rand` and `randint` with reproducible seeding via the T3 `Generator`.

## What Changes

- Add `trainadillo/_random.py` with `rand(*shape, generator=None)` and `randint(low, high, size, *, generator=None)`
- Export both functions from `trainadillo/__init__.py`
- Add tests under `tests/trainadillo/test_random.py`

## Capabilities

### New Capabilities

- `random-creation`: Seeded random tensor factories (`rand`, `randint`) that accept an optional `Generator` and raise when no default generator has been seeded

### Modified Capabilities

(none)

## Non-goals

- Implementing any other random ops (e.g. `randn`, `bernoulli`, `normal`) — only what T4 specifies
- Changing the existing `Generator` or `manual_seed` implementations

## Impact

- New file: `src/trainadillo/_random.py`
- New test file: `tests/trainadillo/test_random.py`
- `src/trainadillo/__init__.py`: add `rand`, `randint` to exports
- No existing code modified; no new external dependencies
