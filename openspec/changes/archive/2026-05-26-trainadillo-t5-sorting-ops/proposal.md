## Why

Trainadillo's sampling path (`_sample()` in bigram.py) requires ranking and
filtering logits — top-k filtering, nucleus (top-p) filtering, and greedy
decoding. These all depend on `topk`, `sort`, and `argmax`, which don't yet
exist. T5 is the next step in the build order after T3 (RNG) and T4 (random
creation functions).

## What Changes

- New `trainadillo/_ops.py` module with `topk` and `sort` functions
- `Tensor.argmax(dim=None)` method added to `trainadillo/_tensor.py`
- `TopKResult` NamedTuple (`values`, `indices`) as the return type for `topk`
- `SortResult` NamedTuple (`values`, `indices`) as the return type for `sort`
- Tests in `tests/trainadillo/test_ops.py`
- Educational doc at `docs/trainadillo/005-sorting-ops.md`

No breaking changes. No changes to existing public API.

## Capabilities

### New Capabilities

- `tensor-sorting-ops`: `topk`, `sort`, and `argmax` operations on Tensors,
  matching the PyTorch interface used in the bigram sampling path.

### Modified Capabilities

- `tensor-foundations`: adds `argmax` method to `Tensor`. This is an additive
  change to an existing capability — no existing requirements change.

## Impact

- `src/trainadillo/_ops.py` — new file
- `src/trainadillo/_tensor.py` — one method added
- `tests/trainadillo/test_ops.py` — new file
- `docs/trainadillo/005-sorting-ops.md` — new file
- No dependency changes; T5 depends only on T1 (already complete)

## Non-goals

- Differentiable versions of these ops (not needed until autograd in T8+)
- GPU/CUDA support
- `argsort` as a standalone function (not used in the sampling path)
- Stable-sort guarantees (numpy's `argsort` uses quicksort by default, which
  is fine for logit filtering)
