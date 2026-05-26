"""Temporary compatibility tests: trainadillo vs PyTorch.

These tests verify that trainadillo's deterministic operations produce results
identical (within float32 precision) to PyTorch. The randomness-only difference
is isolated to multinomial; everything up to that point must match exactly.

TEMPORARY: Delete this file at T20 when PyTorch is removed as a dependency.
"""

import numpy as np
import pytest

from trainadillo._creation import full_like, tensor, zeros_like
from trainadillo._ops import cumsum, softmax, sort, topk
from trainadillo._tensor import Tensor

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _probs_torch(
    logits_np: np.ndarray, temperature: float, top_k: int | None, top_p: float | None
) -> np.ndarray:
    """Run the _sample() filtering pipeline through PyTorch, return probs as numpy."""
    scaled = torch.tensor(logits_np, dtype=torch.float32) / temperature
    if top_k is not None:
        k = min(top_k, scaled.size(-1))
        _, top_indices = torch.topk(scaled, k)
        scaled = torch.full_like(scaled, float("-inf")).scatter(
            0, top_indices, scaled[top_indices]
        )
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        to_remove = (cumulative - sorted_probs) > top_p
        sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
        scaled = torch.zeros_like(scaled).scatter(0, sorted_indices, sorted_logits)
    return torch.softmax(scaled, dim=-1).detach().numpy()


def _probs_ta(
    logits_np: np.ndarray, temperature: float, top_k: int | None, top_p: float | None
) -> np.ndarray:
    """Run the _sample() filtering pipeline through trainadillo, return probs."""
    scaled = tensor(logits_np) / temperature
    if top_k is not None:
        k = min(top_k, scaled.shape[-1])
        _, top_indices = topk(scaled, k)
        scaled = full_like(scaled, float("-inf")).scatter(
            0, top_indices, scaled[top_indices]
        )
    if top_p is not None:
        sorted_logits, sorted_indices = sort(scaled, descending=True)
        sorted_probs = softmax(sorted_logits, dim=-1)
        cumulative = cumsum(sorted_probs, dim=-1)
        to_remove = (cumulative - sorted_probs) > top_p
        sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
        scaled = zeros_like(scaled).scatter(0, sorted_indices, sorted_logits)
    return softmax(scaled, dim=-1).data


# ---------------------------------------------------------------------------
# Exact match for deterministic sub-ops
# ---------------------------------------------------------------------------


def test_softmax_matches_pytorch() -> None:
    data = np.array([1.0, 2.0, 3.0, -100.0, 100.0], dtype=np.float32)
    torch_out = torch.softmax(torch.tensor(data), dim=-1).numpy()
    ta_out = softmax(Tensor(data)).data
    np.testing.assert_allclose(torch_out, ta_out, rtol=1e-5)


def test_masked_fill_matches_pytorch() -> None:
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mask_np = np.array([True, False, True, False])
    torch_out = (
        torch.tensor(data).masked_fill(torch.tensor(mask_np), float("-inf")).numpy()
    )
    ta_out = Tensor(data).masked_fill(Tensor(mask_np), float("-inf")).data
    np.testing.assert_allclose(torch_out, ta_out, rtol=1e-5, equal_nan=True)


def test_scatter_matches_pytorch() -> None:
    base = np.zeros(5, dtype=np.float32)
    idx = np.array([4, 1, 2], dtype=np.int64)
    src = np.array([9.0, 7.0, 5.0], dtype=np.float32)
    torch_out = torch.zeros(5).scatter(0, torch.tensor(idx), torch.tensor(src)).numpy()
    ta_out = Tensor(base).scatter(0, Tensor(idx), Tensor(src)).data
    np.testing.assert_allclose(torch_out, ta_out, rtol=1e-5)


def test_cumsum_matches_pytorch() -> None:
    data = np.array([0.1, 0.3, 0.2, 0.4], dtype=np.float32)
    torch_out = torch.cumsum(torch.tensor(data), dim=-1).numpy()
    ta_out = cumsum(Tensor(data), dim=-1).data
    np.testing.assert_allclose(torch_out, ta_out, rtol=1e-5)


# ---------------------------------------------------------------------------
# Full pipeline probs match
# ---------------------------------------------------------------------------


LOGITS = np.array([0.5, 2.0, -1.0, 3.5, 0.1, 1.2], dtype=np.float32)


@pytest.mark.parametrize(
    ("top_k", "top_p"),
    [
        (2, None),
        (None, 0.9),
        (None, None),
    ],
)
def test_pipeline_probs_match_pytorch(top_k: int | None, top_p: float | None) -> None:
    torch_probs = _probs_torch(LOGITS, temperature=1.0, top_k=top_k, top_p=top_p)
    ta_probs = _probs_ta(LOGITS, temperature=1.0, top_k=top_k, top_p=top_p)
    np.testing.assert_allclose(torch_probs, ta_probs, rtol=1e-5)


# ---------------------------------------------------------------------------
# Corner cases: randomness disappears — exact token agreement
# ---------------------------------------------------------------------------


def test_top_k_1_both_libraries_return_argmax() -> None:
    logits = np.array([0.1, 5.0, 0.2, 0.3], dtype=np.float32)
    torch_probs = _probs_torch(logits, temperature=1.0, top_k=1, top_p=None)
    ta_probs = _probs_ta(logits, temperature=1.0, top_k=1, top_p=None)
    expected = int(np.argmax(logits))
    assert int(np.argmax(torch_probs)) == expected
    assert int(np.argmax(ta_probs)) == expected


def test_top_p_zero_both_libraries_return_argmax() -> None:
    logits = np.array([0.1, 5.0, 0.2, 0.3], dtype=np.float32)
    torch_probs = _probs_torch(logits, temperature=1.0, top_k=None, top_p=0.0)
    ta_probs = _probs_ta(logits, temperature=1.0, top_k=None, top_p=0.0)
    expected = int(np.argmax(logits))
    assert int(np.argmax(torch_probs)) == expected
    assert int(np.argmax(ta_probs)) == expected
