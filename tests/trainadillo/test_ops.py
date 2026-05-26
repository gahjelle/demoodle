"""Tests for trainadillo._ops: topk, sort, softmax, cumsum, multinomial, argmax."""

import numpy as np
import pytest

from trainadillo._ops import (
    SortResult,
    TopKResult,
    cumsum,
    multinomial,
    softmax,
    sort,
    topk,
)
from trainadillo._rng import Generator
from trainadillo._tensor import Tensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tf(*values: float) -> Tensor:
    """Build a 1-D float32 Tensor."""
    return Tensor(np.array(values, dtype=np.float32))


# ---------------------------------------------------------------------------
# topk
# ---------------------------------------------------------------------------


def test_topk_returns_topk_result() -> None:
    result = topk(tf(3.0, 1.0, 4.0, 1.0, 5.0), k=3)
    assert isinstance(result, TopKResult)


def test_topk_named_access() -> None:
    result = topk(tf(3.0, 1.0, 4.0), k=2)
    assert isinstance(result.values, Tensor)
    assert isinstance(result.indices, Tensor)


def test_topk_destructuring() -> None:
    values, indices = topk(tf(3.0, 1.0, 4.0), k=2)
    assert isinstance(values, Tensor)
    assert isinstance(indices, Tensor)


def test_topk_top1_value_and_index() -> None:
    values, indices = topk(tf(3.0, 1.0, 4.0, 1.0, 5.0), k=1)
    assert values.item() == pytest.approx(5.0)
    assert indices.item() == 4


def test_topk_top3_descending_order() -> None:
    values, indices = topk(tf(3.0, 1.0, 4.0, 1.0, 5.0), k=3)
    assert values.tolist() == pytest.approx([5.0, 4.0, 3.0])
    assert indices.tolist() == [4, 2, 0]


def test_topk_k_equals_length_is_full_descending_sort() -> None:
    t = tf(3.0, 1.0, 4.0, 1.0, 5.0)
    values, _ = topk(t, k=5)
    assert values.tolist() == pytest.approx([5.0, 4.0, 3.0, 1.0, 1.0])


def test_topk_indices_recover_values() -> None:
    t = tf(3.0, 1.0, 4.0, 1.0, 5.0)
    values, indices = topk(t, k=3)
    assert t[indices].tolist() == pytest.approx(values.tolist())


# ---------------------------------------------------------------------------
# sort
# ---------------------------------------------------------------------------


def test_sort_returns_sort_result() -> None:
    result = sort(tf(3.0, 1.0, 2.0))
    assert isinstance(result, SortResult)


def test_sort_named_access() -> None:
    result = sort(tf(3.0, 1.0, 2.0))
    assert isinstance(result.values, Tensor)
    assert isinstance(result.indices, Tensor)


def test_sort_ascending() -> None:
    values, indices = sort(tf(3.0, 1.0, 2.0))
    assert values.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert indices.tolist() == [1, 2, 0]


def test_sort_descending() -> None:
    values, indices = sort(tf(3.0, 1.0, 2.0), descending=True)
    assert values.tolist() == pytest.approx([3.0, 2.0, 1.0])
    assert indices.tolist() == [0, 2, 1]


def test_sort_descending_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        sort(
            tf(1.0, 2.0),
            -1,
            True,  # ty: ignore[too-many-positional-arguments] # noqa: FBT003
        )


def test_sort_index_round_trip() -> None:
    t = tf(3.0, 1.0, 4.0, 1.0, 5.0)
    values, indices = sort(t)
    assert t[indices].tolist() == pytest.approx(values.tolist())


def test_sort_index_round_trip_descending() -> None:
    t = tf(3.0, 1.0, 4.0, 1.0, 5.0)
    values, indices = sort(t, descending=True)
    assert t[indices].tolist() == pytest.approx(values.tolist())


# ---------------------------------------------------------------------------
# Tensor.argmax
# ---------------------------------------------------------------------------


def test_argmax_returns_tensor() -> None:
    result = tf(1.0, 3.0, 2.0).argmax()
    assert isinstance(result, Tensor)


def test_argmax_flat_returns_0d_tensor() -> None:
    result = tf(3.0, 1.0, 5.0, 2.0).argmax()
    assert list(result.shape) == []  # 0-D


def test_argmax_flat_correct_index() -> None:
    assert tf(3.0, 1.0, 5.0, 2.0).argmax().item() == 2


def test_argmax_result_is_not_python_int() -> None:
    result = tf(1.0, 2.0, 3.0).argmax()
    assert not isinstance(result, int)
    assert isinstance(result, Tensor)


def test_argmax_with_dim_on_2d() -> None:
    t = Tensor(np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32))
    result = t.argmax(dim=0)
    assert list(result.shape) == [2]
    assert result.tolist() == [1, 0]


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------


def test_softmax_returns_tensor() -> None:
    result = softmax(tf(1.0, 2.0, 3.0))
    assert isinstance(result, Tensor)


def test_softmax_sums_to_one() -> None:
    result = softmax(tf(1.0, 2.0, 3.0))
    assert result.data.sum() == pytest.approx(1.0, abs=1e-6)


def test_softmax_shape_preserved() -> None:
    t = tf(0.5, 1.5, 2.5, 3.5)
    assert list(softmax(t).shape) == [4]


def test_softmax_no_nan_inf_on_large_positive() -> None:
    result = softmax(tf(1000.0, 999.0, 998.0))
    data = result.data
    assert not np.any(np.isnan(data))
    assert not np.any(np.isinf(data))


def test_softmax_no_nan_inf_on_large_negative() -> None:
    result = softmax(tf(-1000.0, -999.0, -998.0))
    data = result.data
    assert not np.any(np.isnan(data))
    assert not np.any(np.isinf(data))


def test_softmax_2d_dim0_each_column_sums_to_one() -> None:
    t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    result = softmax(t, dim=0)
    col_sums = result.data.sum(axis=0)
    assert col_sums == pytest.approx([1.0, 1.0], abs=1e-6)


def test_softmax_2d_dim1_each_row_sums_to_one() -> None:
    t = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    result = softmax(t, dim=1)
    row_sums = result.data.sum(axis=1)
    assert row_sums == pytest.approx([1.0, 1.0], abs=1e-6)


# ---------------------------------------------------------------------------
# cumsum
# ---------------------------------------------------------------------------


def test_cumsum_returns_tensor() -> None:
    result = cumsum(tf(1.0, 2.0, 3.0), dim=-1)
    assert isinstance(result, Tensor)


def test_cumsum_matches_numpy() -> None:
    t = tf(1.0, 2.0, 3.0, 4.0)
    result = cumsum(t, dim=-1)
    expected = np.cumsum(t.data, axis=-1)
    assert result.data == pytest.approx(expected)


def test_cumsum_shape_equals_input_shape() -> None:
    t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    result = cumsum(t, dim=0)
    assert list(result.shape) == [2, 2]


def test_cumsum_2d_along_dim0() -> None:
    t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    result = cumsum(t, dim=0)
    expected = np.array([[1.0, 2.0], [4.0, 6.0]], dtype=np.float32)
    assert result.data == pytest.approx(expected)


# ---------------------------------------------------------------------------
# multinomial
# ---------------------------------------------------------------------------


def _seeded(seed: int) -> Generator:
    return Generator().manual_seed(seed)


def test_multinomial_returns_tensor() -> None:
    probs = softmax(tf(1.0, 2.0, 3.0))
    result = multinomial(probs, 1, generator=_seeded(0))
    assert isinstance(result, Tensor)


def test_multinomial_output_shape() -> None:
    probs = softmax(tf(1.0, 2.0, 3.0, 4.0))
    result = multinomial(probs, 3, generator=_seeded(0))
    assert list(result.shape) == [3]


def test_multinomial_output_dtype_is_int64() -> None:
    probs = softmax(tf(1.0, 2.0, 3.0))
    result = multinomial(probs, 1, generator=_seeded(0))
    assert result.data.dtype == np.int64


def test_multinomial_indices_within_range() -> None:
    probs = softmax(tf(1.0, 2.0, 3.0))
    result = multinomial(probs, 1, generator=_seeded(42))
    assert 0 <= result.item() < 3


def test_multinomial_same_seed_reproducible() -> None:
    probs = softmax(tf(1.0, 2.0, 3.0, 4.0))
    r1 = multinomial(probs, 1, generator=_seeded(99))
    r2 = multinomial(probs, 1, generator=_seeded(99))
    assert r1.item() == r2.item()


def test_multinomial_different_seeds_diverge() -> None:
    probs = softmax(tf(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    results = {multinomial(probs, 1, generator=_seeded(s)).item() for s in range(20)}
    assert len(results) > 1


def test_multinomial_distribution_approximates_probs() -> None:
    # Draw from a strongly skewed distribution; with 10 000 samples the frequencies
    # should be within 0.02 of the true probabilities.
    raw = tf(10.0, 3.0, 1.0)
    probs = softmax(raw)
    true_p = probs.data.tolist()
    n = 10_000
    counts = [0, 0, 0]
    for seed in range(n):
        idx = multinomial(probs, 1, generator=_seeded(seed)).item()
        counts[int(idx)] += 1
    for i, p in enumerate(true_p):
        assert abs(counts[i] / n - p) < 0.02, (
            f"index {i}: expected ~{p:.3f}, got {counts[i] / n:.3f}"
        )
