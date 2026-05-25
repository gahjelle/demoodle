## Why

Trainadillo needs a PyTorch-compatible set of factory functions so that model and
training code can create tensors without importing NumPy directly. T1 delivered the
`Tensor` class; T2 delivers the creation API that makes it usable.

## What Changes

- Add `src/trainadillo/_creation.py` with eight factory functions: `tensor`,
  `zeros`, `ones`, `zeros_like`, `full_like`, `arange`, `stack`, `equal`
- Add `tests/trainadillo/test_creation.py` covering all functions and edge cases
- Add `docs/trainadillo/002-creation-functions.md` explaining design decisions

## Capabilities

### New Capabilities

- `tensor-creation-functions`: Factory functions for constructing `Tensor` objects
  from Python data, shapes, and existing tensors — matching the `torch.*` creation API.

### Modified Capabilities

_(none — T1's `tensor-foundations` spec is not changed by this work)_

## Impact

- New file: `src/trainadillo/_creation.py`
- New file: `tests/trainadillo/test_creation.py`
- `trainadillo/__init__.py` will re-export these functions in T18 (not this item)
- No changes to existing files
- No new dependencies (NumPy already required)

## Non-goals

- Random tensor factories (`rand`, `randint`) — those depend on T3's `Generator`
  and are covered in T4
- `as_tensor()` (non-copying variant of `tensor()`) — not needed yet
- `torch.from_numpy()` — not needed; NumPy is the internal representation
- Wiring into `__init__.py` — deferred to T18
