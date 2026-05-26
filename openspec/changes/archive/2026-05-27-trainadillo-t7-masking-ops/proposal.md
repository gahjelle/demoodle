## Why

The `_sample()` function in `bigram.py` uses `masked_fill` and `scatter_` to implement top-k and nucleus (top-p) filtering, but trainadillo does not yet implement these operations. Without them, the full sampling pipeline cannot run on trainadillo tensors, blocking T19 (bigram integration test) and T20 (PyTorch removal).

## What Changes

- Add `Tensor.masked_fill(mask, value)` — non-in-place method returning a new Tensor with True positions replaced by `value`
- Add `Tensor.scatter(dim, index, src)` — non-in-place method returning a new Tensor with values from `src` written at positions `index` along `dim`
- **BREAKING** (demoodle): Replace the two `scatter_` (in-place) call sites in `bigram.py._sample()` with `scatter` (non-in-place); the in-place `scatter_` form is not implemented in trainadillo
- Add unit tests for both new methods
- Add a temporary PyTorch compatibility test file that verifies the filtering pipeline produces identical results to PyTorch up to float32 precision

## Non-goals

- `masked_fill_` (in-place variant) — not used in the codebase
- `scatter_` (in-place variant) — replaced by `scatter` in this change
- Scalar `src` for `scatter` — only Tensor `src` is needed
- General N-D scatter beyond what numpy fancy indexing supports naturally
- Differentiable versions of these ops — they live in the sampling path, never the gradient path

## Capabilities

### New Capabilities

- `tensor-masking-ops`: `masked_fill` and `scatter` methods on `Tensor`; the non-in-place `scatter` replaces PyTorch's `scatter_` in the demoodle sampling pipeline

### Modified Capabilities

*(none — no existing spec-level requirements change)*

## Impact

- `src/trainadillo/_tensor.py` — two new methods added
- `src/demoodle/architectures/bigram.py` — two `scatter_` call sites rewritten to `scatter`
- `tests/trainadillo/test_ops.py` — unit tests for new methods
- `tests/trainadillo/test_compat_pytorch.py` — new temporary file; deleted at T20
- `tests/architectures/test_bigram.py` — existing tests validate the demoodle rewrite against real PyTorch
