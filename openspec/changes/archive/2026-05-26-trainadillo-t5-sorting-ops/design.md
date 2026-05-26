## Context

T5 adds three ops to trainadillo: `topk`, `sort`, and `argmax`. These are
non-differentiable — they live entirely in the sampling/inference path and will
never appear in a gradient graph. The design is therefore straightforward: pure
numpy wrappers with PyTorch-compatible signatures.

The key upstream context is how these ops are consumed. In bigram's `_sample()`:

```python
top_values, top_indices = torch.topk(logits, k)           # top-k filtering
sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # top-p
token = logits.argmax().item()                            # greedy (temp=0)
```

The indices are not incidental — T7's `scatter_` needs them to reconstruct a
probability distribution over the full vocabulary after filtering.

## Goals / Non-Goals

**Goals:**
- `topk(tensor, k, dim=-1) → TopKResult(values, indices)`
- `sort(tensor, dim=-1, *, descending=False) → SortResult(values, indices)`
- `Tensor.argmax(dim=None) → Tensor`
- Correct named-tuple return types with `.values` and `.indices` access
- Tests covering the cases the sampling pipeline will exercise

**Non-Goals:**
- Differentiable backward implementations (deferred to T10+)
- `argsort` as a standalone function
- Stable-sort guarantees
- GPU support

## Decisions

### D1: Module placement — `_ops.py` (new), `_tensor.py` (augmented)

`topk` and `sort` are standalone functions that take a Tensor as input. They
follow the same pattern as PyTorch's module-level functions and belong in a
dedicated `_ops.py`.

`argmax` is a method: it's called as `logits.argmax()`, not
`trainadillo.argmax(logits)`. Methods that are pure shape/reduction ops live on
`Tensor` directly (alongside `flatten`, `squeeze`, `size`). No separate module
needed.

Considered: putting all three in `_ops.py` and monkey-patching `argmax` onto
`Tensor`. Rejected — monkey-patching is confusing and hard to trace.

### D2: Separate `TopKResult` and `SortResult` NamedTuples

Both functions return `(values, indices)`. Two options:

- **One shared type** (`ValuesAndIndices`) — fewer names, simpler.
- **Two separate types** (`TopKResult`, `SortResult`) — each is its own concept
  with its own semantics; matching PyTorch's `torch.return_types.topk` and
  `torch.return_types.sort`.

We use separate types. `TopKResult.values` are the k largest values in
descending order; `SortResult.values` are fully sorted values. Same fields,
different semantics and different origins. A shared type would obscure that
distinction for readers and for type checkers.

### D3: Descending order via `np.flip` on the index array

`np.argsort` has no `descending` parameter. The two approaches are:

- `np.flip(np.argsort(data, axis=dim), axis=dim)` — flip the *index* array.
  `np.flip` operates on the indices, not the data, so no arithmetic is
  performed on the values. Works for all dtypes. Returns a view (no copy),
  which `np.take_along_axis` and `np.take` handle correctly without requiring
  contiguous memory.
- `np.argsort(-data, axis=dim)` — negate the data before sorting.
  Only correct for floats and signed integers; silently produces wrong results
  for unsigned integer dtypes (`uint8`, `uint16`, etc.) due to wrapping.

We use `np.flip`. It is dtype-agnostic and zero-copy. The concern about
read-only views with negative strides applies to old numpy; modern numpy's
`np.flip` returns a writable view, and the downstream operations
(`np.take_along_axis`, `np.take`) do not require C-contiguous arrays.

### D4: `np.take_along_axis` for gathering values

After computing sorted/top-k indices, we need the corresponding values.
`np.take_along_axis(data, indices, axis=dim)` is numpy's "gather" — it
retrieves values at element-wise index positions along a given axis. This is
the exact semantic we need and handles arbitrary shapes and dimensions correctly.
Alternative: `data[indices]` — works for 1-D but breaks for multi-dimensional
tensors when indices have the same shape as data. We use `take_along_axis`
throughout for correctness.

### D5: `np.take` for selecting first k from sorted indices

To extract the first k elements along an arbitrary `dim` from the descending
index array, `np.take(desc_indices, np.arange(k), axis=dim)` selects k entries
along that axis. For 1-D tensors this is equivalent to `desc_indices[:k]`, but
`np.take` with `axis` handles N-D tensors correctly.

## Risks / Trade-offs

- **`np.argsort` uses quicksort (unstable)** → Ties between equal logit values
  will have non-deterministic index ordering. The sampling path doesn't care
  about tie-breaking order, so this is fine. Stable sort is available via
  `kind="stable"` if ever needed.

## Open Questions

None. All decisions are settled.
