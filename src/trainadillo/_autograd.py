"""Autograd engine: GradFn base class, no_grad context manager, state."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from trainadillo._tensor import Tensor

type NDArray = np.ndarray[Any, np.dtype[np.generic]]


class GradFn(ABC):
    """Base class for all backward functions in the computation graph.

    Each op that supports autograd creates a concrete subclass capturing the
    values needed for its backward computation. The `inputs` list holds the
    Tensor objects whose `.grad` will receive contributions from this node.
    """

    inputs: list[Tensor]

    @abstractmethod
    def backward(self, grad: NDArray) -> list[tuple[Tensor, NDArray]]:
        """Return gradient contributions for each input tensor."""


class _MulGradFn(GradFn):
    """Backward function for element-wise multiplication (a * b).

    Used as the minimal smoke-test GradFn in T8. Differentiable arithmetic
    ops (T10) will define their own GradFns the same way.
    """

    def __init__(
        self,
        inputs: list[Tensor],
        a_data: NDArray,
        b_data: NDArray,
    ) -> None:
        self.inputs = inputs
        # Any: numpy stubs don't resolve arithmetic on generic ndarray types.
        self._a_data: Any = a_data
        self._b_data: Any = b_data

    def backward(self, grad: NDArray) -> list[tuple[Tensor, NDArray]]:
        result = [(self.inputs[0], grad * self._b_data)]  # d/da = grad * b
        if len(self.inputs) > 1:
            result.append((self.inputs[1], grad * self._a_data))  # d/db = grad * a
        return result


class _AutogradState:
    # True by default — gradient tracking is on unless no_grad is active.
    grad_enabled: bool = True


_state = _AutogradState()


class no_grad:  # noqa: N801 — intentionally lowercase to match torch.no_grad()
    """Context manager that disables gradient graph construction.

    While active, all tensor operations produce outputs with grad_fn=None
    regardless of whether inputs have requires_grad=True. Matches
    torch.no_grad() semantics.

    Usage::

        with trainadillo.no_grad():
            output = model(x)   # graph not built
    """

    def __enter__(self) -> None:
        self._prev = _state.grad_enabled
        _state.grad_enabled = False

    def __exit__(self, *args: object) -> None:
        _state.grad_enabled = self._prev


def grad_enabled() -> bool:
    """Return True if gradient computation is currently enabled."""
    return _state.grad_enabled
