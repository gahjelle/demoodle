"""Non-differentiable tensor operations: sorting, reduction, and probability.

These ops live in the sampling and preprocessing path — never in the gradient
path — so they are pure numpy wrappers with no GradFn machinery.
"""

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from trainadillo._rng import resolve_generator
from trainadillo._tensor import Tensor

if TYPE_CHECKING:
    from trainadillo._rng import Generator


class TopKResult(NamedTuple):
    """Return value of topk(): the k largest values and their source indices."""

    values: Tensor
    indices: Tensor


class SortResult(NamedTuple):
    """Return value of sort(): sorted values and the permutation that produced them."""

    values: Tensor
    indices: Tensor


def topk(tensor: Tensor, k: int, dim: int = -1) -> TopKResult:
    """Return the k largest values and their indices along dim.

    Values are returned in descending order (largest first). Indices are the
    positions those values occupied in the original tensor — needed by T7's
    scatter_ to reconstruct a filtered probability distribution.

    Uses argsort (O(n log n)) for simplicity; sampling tensors are small enough
    that the cost is irrelevant compared to model inference.
    """
    data = tensor.data
    asc_indices = np.argsort(data, axis=dim)
    desc_indices = np.flip(asc_indices, axis=dim)
    top_k_indices = np.take(desc_indices, np.arange(k), axis=dim)
    top_k_values = np.take_along_axis(data, top_k_indices, axis=dim)
    return TopKResult(Tensor(top_k_values), Tensor(top_k_indices))


def sort(tensor: Tensor, dim: int = -1, *, descending: bool = False) -> SortResult:
    """Return (sorted_values, sort_indices) along dim.

    Matches torch.sort's return signature. indices[i] is the position in the
    original tensor that produced sorted_values[i] — so tensor[indices] == values
    for 1-D tensors.
    """
    data = tensor.data
    indices = np.argsort(data, axis=dim)
    if descending:
        indices = np.ascontiguousarray(np.flip(indices, axis=dim))
    sorted_values = np.take_along_axis(data, indices, axis=dim)
    return SortResult(Tensor(sorted_values), Tensor(indices))


def softmax(tensor: Tensor, dim: int = -1) -> Tensor:
    """Return a probability distribution over dim using numerically stable softmax.

    Subtracts the per-slice max before exp to prevent overflow/underflow —
    exp(x - max(x)) / sum(exp(x - max(x))). keepdims=True on both reductions
    ensures correct broadcasting for any dim value.

    This is inference-only. The differentiable version lives in nn/functional.py (T15).
    """
    data = tensor.data
    shifted = data - data.max(axis=dim, keepdims=True)
    exp_x = np.exp(shifted)
    return Tensor(exp_x / exp_x.sum(axis=dim, keepdims=True))


def cumsum(tensor: Tensor, dim: int) -> Tensor:
    """Return the cumulative sum along dim."""
    return Tensor(np.cumsum(tensor.data, axis=dim))


def multinomial(
    probs: Tensor,
    num_samples: int,
    *,
    generator: Generator | None = None,
) -> Tensor:
    """Sample num_samples indices from a categorical distribution.

    probs is a 1-D Tensor of non-negative weights (need not sum to 1 — they are
    normalized here). Returns a 1-D int64 Tensor of shape (num_samples,).

    Probs are cast to float64 and renormalized before passing to numpy's choice,
    because float32 softmax output may not sum to exactly 1.0, which numpy requires.
    replace=False matches PyTorch's multinomial default (replacement=False).
    """
    rng = resolve_generator(generator)
    data = probs.data.astype(np.float64)
    p = data / data.sum()
    indices = rng.np_rng.choice(len(p), size=num_samples, replace=False, p=p)
    return Tensor(indices.astype(np.int64))
