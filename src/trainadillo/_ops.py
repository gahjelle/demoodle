"""Non-differentiable tensor operations: sorting and reduction.

These ops live in the sampling and preprocessing path — never in the gradient
path — so they are pure numpy wrappers with no GradFn machinery.
"""

from typing import NamedTuple

import numpy as np

from trainadillo._tensor import Tensor


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
    data = tensor._data  # noqa: SLF001
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
    data = tensor._data  # noqa: SLF001
    indices = np.argsort(data, axis=dim)
    if descending:
        indices = np.ascontiguousarray(np.flip(indices, axis=dim))
    sorted_values = np.take_along_axis(data, indices, axis=dim)
    return SortResult(Tensor(sorted_values), Tensor(indices))
