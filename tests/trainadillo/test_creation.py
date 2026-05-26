"""Tests for trainadillo._creation factory functions."""

import numpy as np
import pytest

from trainadillo._creation import (
    arange,
    equal,
    full_like,
    ones,
    stack,
    tensor,
    zeros,
    zeros_like,
)
from trainadillo._tensor import Tensor, float32, long

# ---------------------------------------------------------------------------
# tensor
# ---------------------------------------------------------------------------


def test_tensor_from_int_list_is_int64() -> None:
    t = tensor([1, 2, 3])
    assert t.dtype == np.dtype(np.int64)
    assert t.tolist() == [1, 2, 3]


def test_tensor_from_float_list_is_float32() -> None:
    # NumPy infers float64 for Python floats; we downgrade to float32.
    t = tensor([1.0, 2.0, 3.0])
    assert t.dtype == np.dtype(np.float32)


def test_tensor_from_float_list_not_float64() -> None:
    t = tensor([1.0, 2.0])
    assert t.dtype != np.dtype(np.float64)


def test_tensor_explicit_dtype_overrides_inference() -> None:
    t = tensor([1, 2, 3], dtype=float32)
    assert t.dtype == np.dtype(np.float32)


def test_tensor_explicit_dtype_long() -> None:
    t = tensor([1.0, 2.0], dtype=long)
    assert t.dtype == np.dtype(np.int64)


def test_tensor_from_scalar() -> None:
    t = tensor(42)
    assert t.item() == 42


def test_tensor_from_numpy_array_copies() -> None:
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = tensor(arr)
    arr[0] = 99.0
    assert t.tolist()[0] == pytest.approx(1.0)


def test_tensor_from_existing_tensor_copies() -> None:
    original = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    copy = tensor(original)
    # Mutate the original's underlying data to verify copy independence
    original.data[0] = 99.0
    assert copy.tolist()[0] == pytest.approx(1.0)


def test_tensor_returns_tensor_instance() -> None:
    assert isinstance(tensor([1, 2]), Tensor)


# ---------------------------------------------------------------------------
# zeros and ones
# ---------------------------------------------------------------------------


def test_zeros_shape_and_dtype() -> None:
    t = zeros(3, 4)
    assert list(t.shape) == [3, 4]
    assert t.dtype == np.dtype(np.float32)
    assert all(v == pytest.approx(0.0) for v in t.flatten().tolist())


def test_zeros_1d() -> None:
    t = zeros(5)
    assert list(t.shape) == [5]
    assert t.dtype == np.dtype(np.float32)


def test_ones_shape_and_dtype() -> None:
    t = ones(2, 3)
    assert list(t.shape) == [2, 3]
    assert t.dtype == np.dtype(np.float32)
    assert all(v == pytest.approx(1.0) for v in t.flatten().tolist())


def test_zeros_not_float64() -> None:
    assert zeros(3).dtype != np.dtype(np.float64)


def test_ones_not_float64() -> None:
    assert ones(3).dtype != np.dtype(np.float64)


# ---------------------------------------------------------------------------
# zeros_like and full_like
# ---------------------------------------------------------------------------


def test_zeros_like_preserves_dtype_float32() -> None:
    src = Tensor(np.ones((2, 3), dtype=np.float32))
    t = zeros_like(src)
    assert t.dtype == np.dtype(np.float32)
    assert list(t.shape) == [2, 3]
    assert all(v == pytest.approx(0.0) for v in t.flatten().tolist())


def test_zeros_like_preserves_dtype_int64() -> None:
    src = Tensor(np.array([1, 2, 3], dtype=np.int64))
    t = zeros_like(src)
    assert t.dtype == np.dtype(np.int64)
    assert t.tolist() == [0, 0, 0]


def test_full_like_preserves_shape_and_dtype() -> None:
    src = Tensor(np.zeros((2, 3), dtype=np.float32))
    t = full_like(src, 7)
    assert list(t.shape) == [2, 3]
    assert t.dtype == np.dtype(np.float32)
    assert all(v == pytest.approx(7.0) for v in t.flatten().tolist())


def test_full_like_int_tensor() -> None:
    src = Tensor(np.zeros((4,), dtype=np.int64))
    t = full_like(src, 5)
    assert t.dtype == np.dtype(np.int64)
    assert t.tolist() == [5, 5, 5, 5]


# ---------------------------------------------------------------------------
# arange
# ---------------------------------------------------------------------------


def test_arange_values() -> None:
    t = arange(5)
    assert t.tolist() == [0, 1, 2, 3, 4]


def test_arange_dtype_is_int64() -> None:
    t = arange(10)
    assert t.dtype == np.dtype(np.int64)


def test_arange_shape() -> None:
    t = arange(7)
    assert list(t.shape) == [7]


# ---------------------------------------------------------------------------
# stack
# ---------------------------------------------------------------------------


def test_stack_default_dim0() -> None:
    a = tensor([1, 2])
    b = tensor([3, 4])
    c = tensor([5, 6])
    result = stack([a, b, c])
    assert list(result.shape) == [3, 2]
    assert result.tolist() == [[1, 2], [3, 4], [5, 6]]


def test_stack_dim1() -> None:
    a = tensor([1, 2])
    b = tensor([3, 4])
    result = stack([a, b], dim=1)
    assert list(result.shape) == [2, 2]
    assert result.tolist() == [[1, 3], [2, 4]]


def test_stack_returns_tensor() -> None:
    result = stack([tensor([1.0, 2.0]), tensor([3.0, 4.0])])
    assert isinstance(result, Tensor)


# ---------------------------------------------------------------------------
# equal
# ---------------------------------------------------------------------------


def test_equal_identical_tensors() -> None:
    a = tensor([1, 2, 3])
    b = tensor([1, 2, 3])
    assert equal(a, b) is True


def test_equal_different_values() -> None:
    a = tensor([1, 2, 3])
    b = tensor([1, 2, 4])
    assert equal(a, b) is False


def test_equal_different_shapes() -> None:
    a = tensor([1, 2])
    b = tensor([1, 2, 3])
    assert equal(a, b) is False


def test_equal_returns_python_bool() -> None:
    result = equal(tensor([1.0]), tensor([1.0]))
    assert type(result) is bool


# ---------------------------------------------------------------------------
# tokens[offsets + context_len] indexing pattern
# ---------------------------------------------------------------------------


def test_tensor_plus_int_as_index() -> None:
    """Verify the bigram.py pattern: tokens[offsets + context_len]."""
    tokens = tensor([10, 20, 30, 40, 50], dtype=long)
    offsets = tensor([0, 1, 2], dtype=long)
    context_len = 1
    result = tokens[offsets + context_len]
    assert isinstance(result, Tensor)
    assert result.tolist() == [20, 30, 40]


def test_tensor_plus_int_produces_tensor() -> None:
    t = tensor([0, 1, 2], dtype=long)
    result = t + 5
    assert isinstance(result, Tensor)
    assert result.dtype == np.dtype(np.int64)
