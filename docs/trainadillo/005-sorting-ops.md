# T5 — Sorting and Reduction Ops

**What T5 builds:** `trainadillo/_ops.py` — `topk` and `sort` as module-level
functions — plus `Tensor.argmax` in `_tensor.py`. All three are non-differentiable
numpy wrappers used in the inference/sampling path.

---

## Why these ops exist

The bigram model's `_sample()` function takes a vector of logits and selects the
next token. Raw logits must be filtered and shaped into a probability distribution
before sampling. Two filtering strategies are standard:

**Top-k filtering:** keep only the k highest-scoring tokens and zero out the
rest. Prevents the model from ever sampling from the long tail of low-probability
tokens.

**Top-p (nucleus) filtering:** keep the smallest set of tokens whose cumulative
probability exceeds p. More adaptive than top-k — when the distribution is
peaked, fewer tokens are kept; when it's flat, more are.

Both strategies require *ranking* the logits:

```python
# top-k: take the k largest logits
values, indices = torch.topk(logits, k)

# top-p: sort all logits descending, walk until cumulative prob > p
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
```

A third strategy — greedy decoding — simply picks the highest-probability token
directly:

```python
token = logits.argmax().item()   # no sampling, always picks the peak
```

These three functions (`topk`, `sort`, `argmax`) are the building blocks for
every sampling strategy in the codebase.

---

## Why indices come back alongside values

Both `topk` and `sort` return `(values, indices)`, not just `values`. The
indices aren't an afterthought — they're often the *primary* result.

Here's why. After filtering with `topk`, you have the k largest logit values.
But to sample from a probability distribution over the *full vocabulary*, you
need to know *which vocabulary positions* those values came from, so you can
construct a sparse probability vector:

```python
# T7 (scatter_) will do this — roughly:
filtered_logits = torch.full_like(logits, float("-inf"))
filtered_logits.scatter_(0, top_k_indices, top_k_values)
# now filtered_logits has -inf everywhere except the top-k positions
```

Without the indices, you can't do the scatter. The values alone are insufficient.

The same applies to top-p: after sorting descending and finding the cutoff point,
you need the `sorted_indices` to scatter the filtered logits back into vocabulary
order before computing probabilities.

---

## Permutations: what argsort actually returns

`np.argsort(arr)` doesn't return the sorted values — it returns the *permutation*
that would sort the array:

```
arr =        [3.0,  1.0,  4.0,  1.0,  5.0]
indices:        0     1     2     3     4

argsort →   [1, 3, 0, 2, 4]   # positions in ascending order of value
            (pos 1 = 1.0 is smallest, pos 4 = 5.0 is largest)
```

Reading it: "to get arr in ascending order, take element at position 1, then 3,
then 0, then 2, then 4." A permutation is a bijection — a reordering.

`np.take_along_axis(arr, indices, axis)` is the "other half": given a
permutation, retrieve the values at those positions. It's numpy's *gather*
operation — the same concept as `torch.gather`.

```
np.take_along_axis(arr, [1, 3, 0, 2, 4], axis=0)
  → [1.0, 1.0, 3.0, 4.0, 5.0]   ✓ sorted ascending
```

`topk` and `sort` are built from these two primitives:

1. `np.argsort` to get the permutation
2. `np.flip` to reverse it if descending (see below)
3. `np.take` to slice the first k (for `topk`)
4. `np.take_along_axis` to retrieve the actual values

---

## Descending order via `np.flip`

`np.argsort` always sorts ascending. For descending order, flip the resulting
index array:

```python
asc_indices = np.argsort(data, axis=dim)   # ascending permutation
desc_indices = np.flip(asc_indices, axis=dim)  # reversed → descending
```

`np.flip` reverses the order of elements along an axis. It returns a *view* with
negative strides — no data is copied, it's just a different way of reading the
same memory. The underlying index values are unchanged; only their traversal
order reverses.

Critically: `np.flip` operates on the *index array*, not the data. No arithmetic
is performed on the values being sorted. This matters because:

```python
# Negation alternative — works for floats and signed ints, but:
np.argsort(-data)   # WRONG for uint8: -(uint8)1 → 255 (overflow)

# Flip alternative — always correct, any dtype
np.flip(np.argsort(data))   # indices are always int64, no overflow possible
```

---

## Separate NamedTuples for TopKResult and SortResult

```python
class TopKResult(NamedTuple):
    values: Tensor
    indices: Tensor

class SortResult(NamedTuple):
    values: Tensor
    indices: Tensor
```

Both have the same fields. Why two types instead of one shared `ValuesAndIndices`?

Because the semantics differ:

- `TopKResult.values` — the k *largest* values, in descending order, a strict
  subset of the original tensor.
- `SortResult.values` — *all* values, sorted. The permutation is total, not partial.

A shared type would lose this distinction. Type checkers and readers would have
no way to know whether they're dealing with a partial top-k extraction or a full
sort. Separate types make the contract explicit — matching the approach PyTorch
takes with `torch.return_types.topk` and `torch.return_types.sort`.

NamedTuples also support both access styles:

```python
# Named access
result = topk(logits, k=50)
print(result.values)
print(result.indices)

# Destructuring
values, indices = topk(logits, k=50)
```

---

## `argmax` — greedy decoding

`argmax` is the degenerate case: no sampling, no temperature, just take the
highest-probability token every time. It's equivalent to `topk(logits, k=1)` but
simpler — it returns a single index, not a tuple.

```python
# temperature = 0 → greedy
token_id = logits.argmax().item()
```

`argmax` returns a `Tensor`, not a Python `int`. This is intentional: returning
a Tensor keeps the API consistent — every operation on a Tensor produces a Tensor.
Call `.item()` explicitly to extract the scalar when you need a Python int.

With `dim=None` (the default), argmax flattens the tensor first and returns a
0-D Tensor containing the flat index. With `dim=n`, it returns a Tensor with
dimension `n` removed, containing the argmax along that axis.

---

## Connection to PyTorch

| trainadillo                | PyTorch                          | Notes                                       |
| -------------------------- | -------------------------------- | ------------------------------------------- |
| `topk(t, k)`               | `torch.topk(t, k)`               | Same signature                              |
| `sort(t, descending=True)` | `torch.sort(t, descending=True)` | Same signature                              |
| `t.argmax()`               | `t.argmax()`                     | Same signature                              |
| `TopKResult`               | `torch.return_types.topk`        | Both NamedTuples with `.values`, `.indices` |
| `SortResult`               | `torch.return_types.sort`        | Both NamedTuples with `.values`, `.indices` |

Where trainadillo simplifies: PyTorch's `topk` and `sort` run optimised C++/CUDA
kernels and handle batch dimensions natively. Trainadillo delegates to numpy's
`argsort`, which is always on CPU and always O(n log n). For the small logit
vectors used in single-token generation (vocabulary size ≤ 65536), this is
entirely adequate.

---

## Summary

| Function     | Returns                       | Key numpy primitive                                    |
| ------------ | ----------------------------- | ------------------------------------------------------ |
| `topk(t, k)` | `TopKResult(values, indices)` | `argsort` + `flip` + `take` + `take_along_axis`        |
| `sort(t)`    | `SortResult(values, indices)` | `argsort` (+ `flip` if descending) + `take_along_axis` |
| `t.argmax()` | 0-D `Tensor`                  | `np.argmax`                                            |
