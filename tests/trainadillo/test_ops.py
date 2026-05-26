"""Tests for trainadillo._ops (topk, sort) and Tensor.argmax."""

import numpy as np
import pytest

from trainadillo._ops import SortResult, TopKResult, sort, topk
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
        sort(tf(1.0, 2.0), -1, True)  # ty: ignore[too-many-positional-arguments]  # noqa: FBT003


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
