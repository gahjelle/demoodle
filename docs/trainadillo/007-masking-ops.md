# 007 — Masking Ops: masked_fill and scatter

This note covers `Tensor.masked_fill` and `Tensor.scatter` — the two operations that
complete the top-k / nucleus (top-p) filtering pipeline in `_sample()`.

---

## masked_fill: why −∞ is the right fill value

`masked_fill(mask, value)` returns a new Tensor where every position where `mask`
is True is replaced with `value`. In sampling, the value is always `float("-inf")`.

Why −∞? The last step of the pipeline is `softmax`. Recall the softmax formula:

```
softmax(x)_i = exp(x_i) / Σ exp(x_j)
```

`exp(-inf) = 0`, so any token whose logit is set to −∞ gets zero probability —
effectively removed from the distribution. This is the cleanest way to zero out
tokens: it integrates perfectly with the numerically stable softmax (which subtracts
the max first) and doesn't require a separate masking step after softmax.

Setting logits to a large negative finite value (e.g. `−1e9`) would work in
practice but is fragile — it depends on the relative magnitude of the logits.
`−inf` is exact.

---

## scatter: writing to positions vs. reading from positions

`scatter(dim, index, src)` returns a new Tensor equal to `self` with values from
`src` written at positions `index` along `dim`. For a 1-D tensor with `dim=0`:

```
result[index[i]] = src[i]   for all i
```

The numpy equivalent is fancy-index assignment on a copy:

```python
out = self.data.copy()
out[index.data] = src.data
return Tensor(out)
```

**Scatter vs. gather** — these are inverses:

- **Gather** (reading): `result[i] = src[index[i]]` — take values *from* specific
  positions in `src` and place them sequentially in `result`. This is what
  `src[index_tensor]` does (standard indexing).
- **Scatter** (writing): `result[index[i]] = src[i]` — take sequential values from
  `src` and place them *at* specific positions in `result`.

In the nucleus filtering pipeline, scatter is used twice:

1. **Top-k masking**: start with a tensor full of −∞, then scatter the top-k logit
   values back to their original positions:
   ```python
   scaled = full_like(scaled, float("-inf")).scatter(0, top_indices, scaled[top_indices])
   ```
   Result: a tensor with −∞ everywhere except the k highest-logit positions.

2. **Unsort after nucleus filtering**: sort descending to compute cumulative
   probabilities, filter low-probability tokens, then scatter back to the original
   token order:
   ```python
   scaled = zeros_like(scaled).scatter(0, sorted_indices, sorted_logits)
   ```
   `sorted_indices` is the sort permutation; scattering by it inverts the sort.
   This works because scatter writes `sorted_logits[i]` to position `sorted_indices[i]`,
   which is exactly the inverse of the sort.

---

## Why non-in-place (scatter, not scatter_)

PyTorch offers both `scatter_` (in-place, modifies `self`) and `scatter` (non-in-place,
returns a new tensor). Trainadillo implements only `scatter`.

The original demoodle code used `scatter_` because the in-place mutation was
convenient — you mutate `mask` in place and then assign `scaled = mask`. But both
call sites can be written as chained non-in-place expressions:

```python
# in-place style (original):
mask = torch.full_like(scaled, float("-inf"))
mask.scatter_(0, top_indices, scaled[top_indices])
scaled = mask

# non-in-place style (current):
scaled = full_like(scaled, float("-inf")).scatter(0, top_indices, scaled[top_indices])
```

The non-in-place form is cleaner (one line instead of three) and avoids mutable
state — a better fit for trainadillo's design. `Tensor.scatter` is also valid PyTorch
(added in 1.8), so this is not a deviation from the PyTorch API.

---

## Compatibility testing strategy

Trainadillo's ops are pure numpy, while PyTorch uses its own kernels. We can't
assume exact bit-for-bit agreement, but we can be precise about *where* randomness
enters the pipeline.

In `_sample()`, only `multinomial` draws random numbers. Everything before it —
`topk`, `sort`, `softmax`, `cumsum`, `masked_fill`, `scatter` — is deterministic.

This gives us a two-layer test strategy:

1. **Exact match for each sub-op**: run identical inputs through both libraries,
   assert `np.allclose(rtol=1e-5)`. These tests prove each building block is correct.

2. **Pipeline match up to multinomial**: replicate the full `_sample()` filtering
   logic in both libraries, stop just before `multinomial`, and assert the probability
   tensors match. This proves the *combined* filtering pipeline is identical — any
   remaining difference between libraries is purely which random number generator
   is used to draw from those probabilities.

Additionally, `top_k=1` and `top_p=0.0` collapse the distribution to a single token
(argmax), eliminating randomness entirely and giving exact agreement even at the
full-pipeline level.

These compatibility tests live in `tests/trainadillo/test_compat_pytorch.py` and
are guarded by `pytest.importorskip("torch")`. They are explicitly temporary:
once PyTorch is removed as a dependency (T20), the file is deleted.
