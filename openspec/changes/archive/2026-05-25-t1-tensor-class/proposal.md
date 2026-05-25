## Why

Trainadillo has package scaffolding but no `Tensor` class. Every other T-item depends on it — creation functions, ops, autograd, modules, the optimizer. T1 builds the data-only layer: a `numpy.ndarray` wrapper with PyTorch's public API but no gradient tracking. Autograd fields and graph construction arrive in T8; T1 is deliberately minimal.

## What Changes

- **New file** `src/trainadillo/_size.py`: `Size(tuple)` — an immutable sequence of ints matching `torch.Size`, including `numel()` and an exact `torch.Size([...])` repr
- **New file** `src/trainadillo/_tensor.py`: `Tensor` class wrapping `np.ndarray`. Module-level dtype constants: `long`, `uint8`, `float32`
- **New file** `tests/trainadillo/test_tensor.py`: unit tests covering all specified behaviours
- **New file** `docs/trainadillo/T1-tensor-class.md`: learning doc — why the wrapper exists, why `__bool__` is absent, how indexing and `view()` work, what T8 will add on top

## Capabilities

### New Capabilities

- `tensor-foundations`: The `Tensor` and `Size` types — shape querying, scalar extraction, type conversion, arithmetic (forward and reflected), comparison, indexing, and repr. No autograd yet.

## Impact

- `src/trainadillo/_size.py` — new file
- `src/trainadillo/_tensor.py` — new file
- `tests/trainadillo/test_tensor.py` — new file
- `docs/trainadillo/T1-tensor-class.md` — new learning doc
- No `demoodle.*` files touched; trainadillo has zero demoodle imports

## Non-goals

- Autograd fields (`grad`, `requires_grad`, `_grad_fn`, `_is_leaf`) — T8
- Creation functions (`zeros`, `ones`, `tensor()`, `arange`, `stack`) — T2
- Random creation functions — T4
- Differentiable arithmetic — T10
- `__init__.py` public API wiring — T18
