"""Tests for trainadillo autograd: Tensor fields, backward(), no_grad, detach."""

import numpy as np
import pytest

from trainadillo._autograd import GradFn, NDArray, _MulGradFn, grad_enabled, no_grad
from trainadillo._tensor import Tensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def scalar(value: float, *, requires_grad: bool = False) -> Tensor:
    return Tensor(np.array(value, dtype=np.float64), requires_grad=requires_grad)


def vec(*values: float, requires_grad: bool = False) -> Tensor:
    return Tensor(np.array(values, dtype=np.float64), requires_grad=requires_grad)


def mul_via_grad_fn(a: Tensor, b: Tensor) -> Tensor:
    """Wire a * b through _MulGradFn, returning a non-leaf result tensor."""
    result = Tensor(a.data * b.data)
    result.grad_fn = _MulGradFn(inputs=[a, b], a_data=a.data, b_data=b.data)
    return result


# ---------------------------------------------------------------------------
# Autograd fields — default state (task 3.4)
# ---------------------------------------------------------------------------


def test_tensor_default_autograd_fields() -> None:
    t = Tensor(np.array([1.0, 2.0]))
    assert t.grad is None
    assert t.requires_grad is False
    assert t.grad_fn is None
    assert t.is_leaf is True


def test_tensor_requires_grad_true() -> None:
    t = Tensor(np.array(1.0), requires_grad=True)
    assert t.requires_grad is True
    assert t.grad_fn is None
    assert t.is_leaf is True


def test_is_leaf_false_when_grad_fn_set() -> None:
    t = Tensor(np.array(1.0))

    class _DummyGradFn(GradFn):
        def __init__(self) -> None:
            self.inputs: list[Tensor] = []

        def backward(self, grad: NDArray) -> list[tuple[Tensor, NDArray]]:  # noqa: ARG002
            return []

    t.grad_fn = _DummyGradFn()
    assert t.is_leaf is False


# ---------------------------------------------------------------------------
# Simple scalar backward (task 4.3)
# ---------------------------------------------------------------------------


def test_backward_simple_mul() -> None:
    x = scalar(3.0, requires_grad=True)
    w = scalar(2.0, requires_grad=True)
    y = mul_via_grad_fn(x, w)

    assert y.grad_fn is not None
    assert len(y.grad_fn.inputs) == 2
    assert y.grad_fn.inputs[0] is x
    assert y.grad_fn.inputs[1] is w

    y.backward()

    assert x.grad is not None
    assert abs(x.grad.item() - 2.0) < 1e-9  # dy/dx = w = 2.0
    assert w.grad is not None
    assert abs(w.grad.item() - 3.0) < 1e-9  # dy/dw = x = 3.0


def test_backward_accumulates_into_existing_grad() -> None:
    x = scalar(3.0, requires_grad=True)
    w = scalar(2.0, requires_grad=True)
    y = mul_via_grad_fn(x, w)
    y.backward()

    # Second backward (simulating gradient accumulation across batches)
    x2 = scalar(1.0, requires_grad=True)
    x2.grad = Tensor(np.array(10.0))  # pre-existing grad
    w2 = scalar(4.0)
    y2 = Tensor(x2.data * w2.data)
    y2.grad_fn = _MulGradFn(inputs=[x2, w2], a_data=x2.data, b_data=w2.data)
    y2.backward()

    assert abs(x2.grad.item() - 14.0) < 1e-9  # 10 + 4


def test_backward_no_grad_when_requires_grad_false() -> None:
    x = scalar(3.0, requires_grad=False)
    w = scalar(2.0, requires_grad=True)
    y = mul_via_grad_fn(x, w)
    y.backward()

    assert x.grad is None  # requires_grad=False → no accumulation
    assert w.grad is not None
    assert abs(w.grad.item() - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# Diamond graph — gradient accumulates, not overwrites (task 4.3)
# ---------------------------------------------------------------------------


class _AddGradFn(GradFn):
    """Backward for a + b: both inputs receive the upstream gradient."""

    def __init__(self, inputs: list[Tensor]) -> None:
        self.inputs = inputs

    def backward(self, grad: NDArray) -> list[tuple[Tensor, NDArray]]:
        return [(self.inputs[0], grad), (self.inputs[1], grad)]


def test_backward_diamond_accumulates() -> None:
    # loss = x*2 + x*3  →  d_loss/d_x = 2 + 3 = 5
    x = scalar(1.0, requires_grad=True)

    a = Tensor(x.data * 2.0)
    a.grad_fn = _MulGradFn(inputs=[x], a_data=x.data, b_data=np.array(2.0))

    b = Tensor(x.data * 3.0)
    b.grad_fn = _MulGradFn(inputs=[x], a_data=x.data, b_data=np.array(3.0))

    loss = Tensor(a.data + b.data)
    loss.grad_fn = _AddGradFn(inputs=[a, b])

    loss.backward()

    assert x.grad is not None
    assert abs(x.grad.item() - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# backward() raises on non-scalar (task 4.3)
# ---------------------------------------------------------------------------


def test_backward_raises_on_non_scalar() -> None:
    t = vec(1.0, 2.0, requires_grad=True)
    with pytest.raises(RuntimeError):
        t.backward()


# ---------------------------------------------------------------------------
# Leaf tensor with requires_grad=True and no grad_fn
# ---------------------------------------------------------------------------


def test_backward_leaf_no_grad_fn() -> None:
    x = scalar(5.0, requires_grad=True)
    x.backward()
    assert x.grad is not None
    assert abs(x.grad.item() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# no_grad context manager (task 4.3)
# ---------------------------------------------------------------------------


def test_no_grad_disables_grad_tracking() -> None:
    assert grad_enabled() is True
    with no_grad():
        assert grad_enabled() is False
    assert grad_enabled() is True


@pytest.mark.skip(reason="T10: ops must check grad_enabled() before building GradFn")
def test_no_grad_ops_produce_no_grad_fn() -> None:
    # GIVEN a tensor with requires_grad=True
    # WHEN an op is performed inside no_grad
    # THEN the result has grad_fn=None and requires_grad=False
    # This scenario cannot be tested until T10 wires grad_enabled() into arithmetic ops.
    x = scalar(3.0, requires_grad=True)
    with no_grad():
        y = x * 2  # noqa: F841  # once T10 lands, y.grad_fn must be None here


def test_no_grad_restores_on_exception() -> None:
    msg = "simulated error"
    with pytest.raises(ValueError, match=msg), no_grad():
        raise ValueError(msg)
    assert grad_enabled() is True


def test_no_grad_nested() -> None:
    with no_grad():
        assert grad_enabled() is False
        with no_grad():
            assert grad_enabled() is False
        assert grad_enabled() is False
    assert grad_enabled() is True


# ---------------------------------------------------------------------------
# detach() (task 4.3)
# ---------------------------------------------------------------------------


def test_detach_clears_grad_fn() -> None:
    x = scalar(3.0, requires_grad=True)
    w = scalar(2.0)
    y = mul_via_grad_fn(x, w)
    assert y.grad_fn is not None

    z = y.detach()
    assert z.grad_fn is None
    assert z.requires_grad is False


def test_detach_shares_data() -> None:
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    z = x.detach()
    assert z.data is x.data  # same numpy array object
