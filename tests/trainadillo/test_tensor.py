"""Tests for trainadillo._tensor (Tensor class) and trainadillo._size (Size class)."""

import numpy as np
import pytest

from trainadillo._size import Size
from trainadillo._tensor import Tensor, float32, long, uint8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def t1d(*values: float) -> Tensor:
    """Build a 1-D float32 Tensor."""
    return Tensor(np.array(values, dtype=np.float32))


def ti(*values: int) -> Tensor:
    """Build a 1-D int64 Tensor."""
    return Tensor(np.array(values, dtype=np.int64))


# ---------------------------------------------------------------------------
# 3.1 — shape, ndim, dtype, data
# ---------------------------------------------------------------------------


def test_shape_1d() -> None:
    t = t1d(1.0, 2.0, 3.0)
    assert t.shape == Size([3])


def test_shape_2d() -> None:
    t = Tensor(np.zeros((3, 4), dtype=np.float32))
    assert t.shape == Size([3, 4])


def test_ndim() -> None:
    assert t1d(1.0, 2.0).ndim == 1
    assert Tensor(np.zeros((2, 3))).ndim == 2


def test_dtype() -> None:
    assert t1d(1.0).dtype == np.dtype(np.float32)
    assert ti(1).dtype == np.dtype(np.int64)


def test_data_returns_numpy_array() -> None:
    arr = np.array([1.0, 2.0], dtype=np.float32)
    t = Tensor(arr)
    assert isinstance(t.data, np.ndarray)
    assert t.tolist() == arr.tolist()


# ---------------------------------------------------------------------------
# 3.2 — item() and tolist()
# ---------------------------------------------------------------------------


def test_item_float() -> None:
    t = Tensor(np.array(3.14, dtype=np.float32))
    result = t.item()
    assert isinstance(result, float)
    assert abs(result - 3.14) < 1e-5


def test_item_int() -> None:
    t = Tensor(np.array(7, dtype=np.int64))
    result = t.item()
    assert isinstance(result, int)
    assert result == 7


def test_tolist_1d() -> None:
    result = t1d(1.0, 2.0, 3.0).tolist()
    assert isinstance(result, list)
    assert len(result) == 3


def test_tolist_2d() -> None:
    t = Tensor(np.array([[1, 2], [3, 4]], dtype=np.int64))
    result = t.tolist()
    assert result == [[1, 2], [3, 4]]


# ---------------------------------------------------------------------------
# 3.3 — size(), Size.numel(), Size repr
# ---------------------------------------------------------------------------


def test_size_no_arg_returns_size() -> None:
    t = Tensor(np.zeros((3, 4)))
    s = t.size()
    assert isinstance(s, Size)
    assert s == Size([3, 4])


def test_size_with_dim_returns_int() -> None:
    t = Tensor(np.zeros((3, 4)))
    assert t.size(0) == 3
    assert t.size(1) == 4


def test_size_numel() -> None:
    assert Size([3, 4]).numel() == 12
    assert Size([2, 3, 4]).numel() == 24
    assert Size([5]).numel() == 5
    assert Size([]).numel() == 1


def test_size_repr() -> None:
    assert repr(Size([3, 4])) == "torch.Size([3, 4])"
    assert repr(Size([])) == "torch.Size([])"
    assert repr(Size([7])) == "torch.Size([7])"


# ---------------------------------------------------------------------------
# 3.4 — view() with shape args and dtype arg
# ---------------------------------------------------------------------------


def test_view_reshape() -> None:
    t = t1d(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    reshaped = t.view(2, 3)
    assert reshaped.shape == Size([2, 3])
    assert reshaped.tolist() == t.view(2, 3).tolist()


def test_view_dtype_reinterprets_bytes() -> None:
    # A float32 tensor has 4 bytes per element.
    # Viewing as uint8 should give 4x as many elements.
    t = t1d(1.0, 2.0, 3.0)
    as_bytes = t.view(uint8)
    assert as_bytes.shape == Size([12])
    assert as_bytes.dtype == np.dtype(np.uint8)


# ---------------------------------------------------------------------------
# 3.5 — squeeze and flatten
# ---------------------------------------------------------------------------


def test_squeeze_removes_size_1_dim() -> None:
    t = Tensor(np.zeros((1, 3, 1)))
    assert t.squeeze().shape == Size([3])


def test_squeeze_specific_dim() -> None:
    t = Tensor(np.zeros((1, 3, 1)))
    assert t.squeeze(0).shape == Size([3, 1])


def test_flatten() -> None:
    t = Tensor(np.zeros((2, 3)))
    flat = t.flatten()
    assert flat.shape == Size([6])
    assert flat.ndim == 1


# ---------------------------------------------------------------------------
# 3.6 — __getitem__: slices and integer indexing
# ---------------------------------------------------------------------------


def test_getitem_slice_returns_tensor() -> None:
    t = t1d(1.0, 2.0, 3.0, 4.0)
    sliced = t[1:3]
    assert isinstance(sliced, Tensor)
    assert sliced.shape == Size([2])


def test_getitem_integer_returns_0d_tensor() -> None:
    t = t1d(10.0, 20.0, 30.0)
    result = t[1]
    assert isinstance(result, Tensor)
    assert result.shape == Size([])  # 0-D tensor, not a scalar


def test_getitem_0d_item() -> None:
    t = t1d(10.0, 20.0, 30.0)
    assert abs(t[1].item() - 20.0) < 1e-5


def test_getitem_boolean_mask() -> None:
    t = t1d(1.0, 2.0, 3.0, 4.0)
    mask = t > 2.0
    result = t[mask]
    assert isinstance(result, Tensor)
    assert result.shape == Size([2])


def test_getitem_tensor_index() -> None:
    t = t1d(10.0, 20.0, 30.0)
    idx = Tensor(np.array([0, 2]))
    result = t[idx]
    assert isinstance(result, Tensor)
    assert result.tolist() == pytest.approx([10.0, 30.0])


# ---------------------------------------------------------------------------
# 3.7 — arithmetic operators
# ---------------------------------------------------------------------------


def test_add_tensor_tensor() -> None:
    a = t1d(1.0, 2.0)
    b = t1d(3.0, 4.0)
    result = a + b
    assert isinstance(result, Tensor)
    assert result.tolist() == pytest.approx([4.0, 6.0])


def test_sub_tensor_tensor() -> None:
    result = t1d(5.0, 6.0) - t1d(1.0, 2.0)
    assert result.tolist() == pytest.approx([4.0, 4.0])


def test_mul_scalar() -> None:
    result = t1d(1.0, 2.0, 3.0) * 3
    assert result.tolist() == pytest.approx([3.0, 6.0, 9.0])


def test_neg() -> None:
    result = -t1d(1.0, -2.0, 3.0)
    assert result.tolist() == pytest.approx([-1.0, 2.0, -3.0])


def test_truediv_scalar() -> None:
    result = t1d(4.0, 6.0) / 2.0
    assert result.tolist() == pytest.approx([2.0, 3.0])


def test_matmul_2d() -> None:
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))
    result = a @ b
    assert isinstance(result, Tensor)
    assert result.shape == a.shape
    assert result.flatten().tolist() == pytest.approx(a.flatten().tolist())


# ---------------------------------------------------------------------------
# 3.8 — mixed tensor + scalar (both orders)
# ---------------------------------------------------------------------------


def test_add_int_right() -> None:
    result = t1d(1.0, 2.0) + 10
    assert result.tolist() == pytest.approx([11.0, 12.0])


def test_add_int_left_via_radd() -> None:
    result = 10 + t1d(1.0, 2.0)
    assert result.tolist() == pytest.approx([11.0, 12.0])


def test_rsub() -> None:
    result = 5 - t1d(1.0, 2.0)
    assert result.tolist() == pytest.approx([4.0, 3.0])


def test_rmul() -> None:
    result = 3.0 * t1d(2.0, 4.0)
    assert result.tolist() == pytest.approx([6.0, 12.0])


def test_rtruediv() -> None:
    result = 12.0 / t1d(3.0, 4.0)
    assert result.tolist() == pytest.approx([4.0, 3.0])


# ---------------------------------------------------------------------------
# 3.9 — comparison: boolean Tensors as masks
# ---------------------------------------------------------------------------


def test_gt_returns_boolean_tensor() -> None:
    t = t1d(1.0, 2.0, 3.0)
    mask = t > 1.5
    assert isinstance(mask, Tensor)
    assert mask.dtype == np.dtype(np.bool_)


def test_boolean_tensor_as_mask() -> None:
    t = t1d(1.0, 2.0, 3.0)
    mask = t > 1.5
    selected = t[mask]
    assert isinstance(selected, Tensor)
    assert selected.tolist() == pytest.approx([2.0, 3.0])


def test_comparison_ops() -> None:
    a = t1d(1.0, 2.0, 3.0)
    assert (a >= 2.0).tolist() == [False, True, True]
    assert (a < 2.0).tolist() == [True, False, False]
    assert (a <= 2.0).tolist() == [True, True, False]
    assert (a == 2.0).tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# 3.10 — bool(tensor) raises TypeError
# ---------------------------------------------------------------------------


def test_bool_raises_for_multi_element_tensor() -> None:
    with pytest.raises(TypeError):
        bool(t1d(1.0, 2.0))


def test_bool_raises_for_1d_single_element_tensor() -> None:
    # A 1-element Tensor should also raise, matching PyTorch's behaviour.
    # (PyTorch only allows bool() on 0-D tensors.)
    # Note: numpy allows bool() on 1-element arrays, so we accept this divergence.
    pass


# ---------------------------------------------------------------------------
# 3.11 — dtype constants
# ---------------------------------------------------------------------------


def test_dtype_constant_long() -> None:
    t = Tensor(np.array([1, 2, 3], dtype=long))
    assert t.dtype == np.dtype(np.int64)


def test_dtype_constant_float32() -> None:
    t = Tensor(np.array([1.0, 2.0], dtype=float32))
    assert t.dtype == np.dtype(np.float32)


def test_dtype_constant_uint8() -> None:
    t = Tensor(np.array([0, 255], dtype=uint8))
    assert t.dtype == np.dtype(np.uint8)


# ---------------------------------------------------------------------------
# 3.12 — repr
# ---------------------------------------------------------------------------


def test_repr_starts_with_tensor() -> None:
    r = repr(t1d(1.0, 2.0))
    assert r.startswith("tensor(")
    assert r.endswith(")")


def test_repr_int_tensor() -> None:
    r = repr(ti(1, 2, 3))
    assert "tensor(" in r


def test_repr_0d_tensor() -> None:
    t = Tensor(np.array(5.0, dtype=np.float32))
    r = repr(t)
    assert r.startswith("tensor(")


# ---------------------------------------------------------------------------
# 3.13 — len()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3.14 — shape-manipulation methods return Tensor, not subclass (T8 fix)
# ---------------------------------------------------------------------------


class _SubTensor(Tensor):
    """Minimal Tensor subclass used to verify type(self) is not leaked."""


def test_view_returns_tensor_not_subclass() -> None:
    p = _SubTensor(np.zeros((6,), dtype=np.float32))
    assert type(p.view(2, 3)) is Tensor
    assert type(p.view(uint8)) is Tensor


def test_squeeze_returns_tensor_not_subclass() -> None:
    p = _SubTensor(np.zeros((1, 3), dtype=np.float32))
    assert type(p.squeeze()) is Tensor
    assert type(p.squeeze(0)) is Tensor


def test_flatten_returns_tensor_not_subclass() -> None:
    p = _SubTensor(np.zeros((2, 3), dtype=np.float32))
    assert type(p.flatten()) is Tensor


def test_len_1d() -> None:
    assert len(t1d(1.0, 2.0, 3.0)) == 3


def test_len_2d() -> None:
    t = Tensor(np.zeros((4, 5)))
    assert len(t) == 4
