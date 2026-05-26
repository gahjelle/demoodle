# 006 — Probability Ops: softmax, cumsum, multinomial

These three functions are the probability-distribution layer of the sampling pipeline.
They are non-differentiable inference utilities, not training ops — they live in
`_ops.py` alongside the sorting functions from T5.

---

## Why these three?

Look at `_sample()` in `bigram.py`:

```python
# top-p (nucleus) filtering
sorted_probs = softmax(sorted_logits, dim=-1)
cumulative = cumsum(sorted_probs, dim=-1)
to_remove = (cumulative - sorted_probs) > top_p

# final draw
probs = softmax(scaled, dim=-1)
return multinomial(probs, num_samples=1, generator=generator).squeeze(0)
```

`softmax` converts raw logit scores into a probability distribution. `cumsum` turns
that distribution into a cumulative mass function used by nucleus filtering. `multinomial`
draws a sample from the final distribution.

---

## softmax — why numerical stability matters

The naive formula is `exp(x) / sum(exp(x))`. This overflows for large `x` (e.g. logits
of 1000) and underflows to all-zeros for very negative `x`. Both produce NaN or Inf.

The fix: subtract the maximum before exponentiation.

```
softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
```

Subtracting `max(x)` shifts the largest element to 0, so `exp(0) = 1`. All other
values are ≤ 1. The result is identical mathematically — the constant cancels:

```
exp(x_i - c) / sum_j(exp(x_j - c))
= exp(x_i) * exp(-c) / (exp(-c) * sum_j(exp(x_j)))
= exp(x_i) / sum_j(exp(x_j))
```

The numpy implementation uses `keepdims=True` on both the max and sum reductions. This
keeps a broadcastable shape so the subtraction and division work correctly for any `dim`
without reshaping:

```python
shifted = data - data.max(axis=dim, keepdims=True)
exp_x = np.exp(shifted)
return exp_x / exp_x.sum(axis=dim, keepdims=True)
```

PyTorch's C++ softmax does the same stabilization, plus a fused CUDA kernel. We just
call numpy — functionally identical for small vocabularies.

This version is **inference-only**. The differentiable version (T15) lives in
`nn/functional.py` and creates a `GradFn` node so gradients can flow backward through
the training loss.

---

## cumsum — nucleus filtering in action

`cumsum(probs, dim=-1)` converts `[0.4, 0.3, 0.2, 0.1]` into `[0.4, 0.7, 0.9, 1.0]`.
It's a running total — numpy's `np.cumsum`.

In nucleus (top-p) sampling, `sorted_probs` is sorted in descending order. The cumulative
sum grows from left to right. We remove tokens once the cumulative mass exceeds `top_p`:

```python
to_remove = (cumulative - sorted_probs) > top_p
```

Subtracting `sorted_probs` gives the cumulative mass *before* including that token.
Tokens are kept until adding them would push the total past `top_p`. This is the "smallest
set of tokens whose cumulative probability exceeds p" definition from the original nucleus
sampling paper.

---

## multinomial — sampling from a categorical distribution

`multinomial(probs, num_samples)` draws `num_samples` indices without replacement,
where index `i` is selected with probability `probs[i]`.

Under the hood: `numpy.random.Generator.choice(n, size, replace=False, p=probs)`.

### The float32 normalization problem

`np.random.Generator.choice` requires `p` to sum to exactly 1.0. Our `probs` come from
`softmax`, which operates in float32. Float32 has ~7 significant decimal digits, so a
sum of 50 000 probabilities (the GPT-2 vocabulary size) can be off by ~0.0001. numpy
treats this as a violated constraint and raises `ValueError`.

The fix: cast to float64 before normalizing.

```python
p = probs._data.astype(np.float64)
p = p / p.sum()
```

Float64 has ~15 significant digits — enough precision for numpy's tolerance check.
The cast is local to `multinomial` and never appears in the returned Tensor (indices
are always int64).

### Reproducibility via Generator

Passing `generator=g` makes draws deterministic: `g._np_rng.choice(...)`. The same
generator state produces the same index every time. This is how the demoodle CLI
produces reproducible generations from a given seed.

### PyTorch comparison

`torch.multinomial(probs, num_samples, replacement=False)` — same semantics.
PyTorch's implementation dispatches to CUDA for GPU tensors or its C++ Philox RNG
for CPU tensors. Trainadillo uses numpy's PCG64. The interface is identical; the
RNG algorithm differs, so given the same seed the actual samples will differ between
torch and trainadillo.
