## Context

`_sample()` in `bigram.py` filters a logit distribution using top-k and nucleus (top-p) strategies before sampling. Both strategies rely on two tensor operations not yet in trainadillo:

- `masked_fill(mask, value)` — replaces positions where a boolean mask is True with a fill value (used to zero out low-probability tokens by setting their logits to `-inf`)
- `scatter_` (in-place) — writes values from a source tensor into specific index positions (used to apply the top-k mask and to unsort logits after nucleus filtering)

PyTorch offers both in-place (`scatter_`) and non-in-place (`scatter`) variants. The codebase currently uses `scatter_`, but the two call sites can each be rewritten as a single chained expression using the non-in-place `scatter`, eliminating the need for mutation.

## Goals / Non-Goals

**Goals:**
- Implement `masked_fill` and `scatter` as non-in-place Tensor methods
- Rewrite `bigram.py._sample()` to use `scatter` instead of `scatter_`
- Validate the demoodle rewrite against real PyTorch before implementing the trainadillo version
- Prove the filtering pipeline is bit-for-bit equivalent to PyTorch via compatibility tests

**Non-Goals:**
- `masked_fill_` or `scatter_` (in-place variants)
- Scalar `src` for `scatter`
- Differentiable variants (these ops are in the sampling path only)

## Decisions

### Decision 1: `scatter` (non-in-place) instead of `scatter_` (in-place)

The two `scatter_` call sites in `_sample()` both discard or immediately rebind the result, making in-place mutation the only reason they work. Both can be rewritten as chained non-in-place expressions:

```python
# Before
mask = torch.full_like(scaled, float("-inf"))
mask.scatter_(0, top_indices, scaled[top_indices])
scaled = mask

# After
scaled = torch.full_like(scaled, float("-inf")).scatter(0, top_indices, scaled[top_indices])
```

The non-in-place `Tensor.scatter` exists in PyTorch (added 1.8) so this is not a deviation from the PyTorch API. It avoids implementing mutable state in trainadillo for a case where immutability is straightforward.

### Decision 2: Methods go in `_tensor.py`, not `_ops.py`

`masked_fill` and `scatter` are instance methods invoked on a Tensor receiver. The project already places `argmax` (also a method) directly in `_tensor.py` rather than `_ops.py`. Standalone functions that return named tuples (`topk`, `sort`) live in `_ops.py`. This distinction is maintained here.

### Decision 3: `scatter` numpy implementation via copy + fancy indexing

For the 1-D dim=0 case that covers all current usage:

```python
out = self._data.copy()
out[index._data] = src._data
return Tensor(out)
```

`np.put_along_axis` handles the general N-D case but has subtly different shape requirements (index must broadcast with `arr`). Since only 1-D dim=0 is needed and the simpler implementation is correct for that case, the fancy-index-on-copy approach is used. The general case is not needed until transformer attention (T28), where it can be revisited.

### Decision 4: Demoodle update first, PyTorch validation second

The `bigram.py` rewrite is performed and validated against real PyTorch before any trainadillo code is written. This ensures the non-in-place rewrite is correct before it becomes the target spec for the trainadillo implementation.

### Decision 5: Temporary PyTorch compatibility test file

A dedicated `tests/trainadillo/test_compat_pytorch.py` (guarded by `pytest.importorskip("torch")`) tests that the filtering pipeline up to `multinomial` produces identical `probs` tensors in both libraries. This file is explicitly temporary — it will be deleted at T20 when PyTorch is removed. Marking it separately makes it easy to find and delete.

The test strategy has two layers:
1. **Exact match** for deterministic sub-ops (`softmax`, `scatter`, `masked_fill`, `cumsum`) given identical inputs
2. **Pipeline match**: replicate `_sample()` logic up to (not including) `multinomial`, assert `probs` tensors match — proves the filtering is identical; only RNG choice can differ

Corner cases where randomness disappears (`top_k=1`, `top_p=0.0` → both collapse to argmax) give exact agreement even at the full-pipeline level.

## Risks / Trade-offs

- **Rewrite introduces a PyTorch API divergence** — `scatter` (non-in-place) is valid PyTorch but the original code used `scatter_`. Any future code copied from PyTorch examples may use `scatter_` and need a similar rewrite. → Mitigation: document the convention in `CONTEXT.md` or a comment in `bigram.py`.
- **`scatter` fancy-indexing-on-copy doesn't generalise** — when T28 needs N-D scatter, the implementation will need to be extended. → Mitigation: the current implementation is correct for all actual usage; T28 can extend it then.
- **Compatibility tests depend on PyTorch being installed** — if T20 is implemented before all compat tests are written, the tests can never run. → Mitigation: the tests are written as part of T7, well before T20.

## Open Questions

None — all design decisions are resolved.
