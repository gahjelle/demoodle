## Why

The sampling pipeline in `bigram.py` requires `softmax`, `cumsum`, and `multinomial` to implement temperature scaling, top-k, and nucleus (top-p) filtering. These are currently provided by PyTorch; T6 implements them in trainadillo so the bigram sampling path runs without torch.

## What Changes

- Add `softmax(tensor, dim=-1)` to `_ops.py` — numerically stable inference-only softmax
- Add `cumsum(tensor, dim)` to `_ops.py` — cumulative sum along a dimension
- Add `multinomial(probs, num_samples, *, generator=None)` to `_ops.py` — categorical sampling
- Move `_resolve_generator` helper from `_random.py` to `_rng.py` so `multinomial` can share it without circular imports

## Capabilities

### New Capabilities

- `probability-ops`: Probability distribution operations (`softmax`, `cumsum`, `multinomial`) for the sampling pipeline

### Modified Capabilities

- `rng`: `_resolve_generator` moves from `_random.py` to `_rng.py` — no public API change, internal refactor

## Impact

- `src/trainadillo/_rng.py` — gains `_resolve_generator`
- `src/trainadillo/_random.py` — imports `_resolve_generator` from `_rng` instead of defining it
- `src/trainadillo/_ops.py` — gains `softmax`, `cumsum`, `multinomial`
- `tests/trainadillo/test_ops.py` — gains tests for all three functions
- No public API changes; T18 will wire exports into `__init__.py`

## Non-goals

- Differentiable softmax (that's T15 in `nn/functional.py`)
- Multi-dimensional multinomial (1-D probs only, matching the sampling use case)
- Exporting these functions from `trainadillo.__init__` (that's T18)
