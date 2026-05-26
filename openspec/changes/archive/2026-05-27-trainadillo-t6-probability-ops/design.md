## Context

T6 completes the non-differentiable ops needed by the bigram sampling pipeline (`_sample()` in `bigram.py`). T5 added `topk`, `sort`, and `argmax`. T6 adds the probability distribution ops: `softmax`, `cumsum`, `multinomial`. All three are inference-only — they never appear in the gradient path, so no `GradFn` machinery is needed.

The `_resolve_generator` helper in `_random.py` (which resolves an explicit generator or falls back to the module-level default) is also needed by `multinomial`. Moving it to `_rng.py` avoids either duplicating the logic or creating a conceptually awkward import direction.

## Goals / Non-Goals

**Goals:**
- Implement numerically stable `softmax` for inference
- Implement `cumsum` as a thin numpy wrapper
- Implement `multinomial` with generator support for reproducible sampling
- Refactor `_resolve_generator` to `_rng.py` to make it shareable

**Non-Goals:**
- Differentiable softmax (T15, `nn/functional.py`)
- Multi-dimensional probs in `multinomial`
- Exporting from `trainadillo.__init__` (T18)

## Decisions

### `softmax` implementation: numerically stable, axis-aware

Use the log-sum-exp stabilization: subtract the per-slice max before exp, then divide by the per-slice sum. Both reduction ops use `keepdims=True` so the result broadcasts correctly for any `dim`:

```python
shifted = data - data.max(axis=dim, keepdims=True)
exp_x = np.exp(shifted)
return Tensor(exp_x / exp_x.sum(axis=dim, keepdims=True))
```

**Alternative considered:** unnormalized `np.exp(x) / np.exp(x).sum()`. Rejected — overflows for large logits (common with untrained models).

### `multinomial` probability normalization

`np.random.Generator.choice` requires the `p` array to sum exactly to 1.0. Float32 softmax output may not satisfy this due to rounding. Fix: normalize before passing:

```python
p = probs_data.astype(np.float64)
p = p / p.sum()
rng._np_rng.choice(len(p), size=num_samples, replace=False, p=p)
```

Cast to float64 first, then normalize — this gives numpy enough precision to satisfy the sum constraint. `replace=False` matches PyTorch's default `replacement=False` semantics.

**Alternative considered:** skipping normalization and letting numpy raise. Rejected — silent failures from rounding are worse than a defensive normalize.

### `num_samples` as positional parameter

Signature: `multinomial(probs, num_samples, *, generator=None)`. Matches torch's positional usage in `bigram.py`: `torch.multinomial(probs, num_samples=1, generator=generator)`.

### `_resolve_generator` moved to `_rng.py`

`_rng.py` defines the `Generator` class and the `_default_generator` it guards. `_resolve_generator` is fundamentally about that guard — it belongs there. `_random.py` becomes an importer rather than a definer.

**Import direction after refactor:**
```
_rng.py        defines Generator, _default_generator, _resolve_generator
_random.py     imports _resolve_generator from _rng
_ops.py        imports _resolve_generator from _rng
```

No circular imports — `_ops.py` and `_random.py` both depend on `_rng.py`, which depends on nothing in trainadillo.

### All three functions in `_ops.py`

The plan specifies `_ops.py`. Keeps the non-differentiable ops in one place. T5's sorting ops and T6's probability ops are both sampling-pipeline utilities.

## Risks / Trade-offs

- **Statistical correctness of multinomial** — hard to unit-test exactly; use a large-N frequency test (e.g. 10 000 draws, assert frequencies are within tolerance of input probs). The test is probabilistic but with large N it's effectively deterministic.
- **float64 cast in multinomial** — probs are float32 internally but cast to float64 for numpy's choice. The cast is local to `multinomial` and doesn't affect the returned Tensor's dtype (indices are int64 regardless).

## Open Questions

None. All design decisions made during exploration.
