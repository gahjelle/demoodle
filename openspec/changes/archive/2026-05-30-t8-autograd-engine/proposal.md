## Why

Trainadillo currently wraps numpy for tensor operations but has no way to compute
gradients — every op is a dead end. T8 wires the electricity: a computation graph
built during the forward pass and walked in reverse during backward, enabling
the training loop that all subsequent milestones depend on.

## What Changes

- Add `grad`, `requires_grad`, `grad_fn` fields and `is_leaf` property to `Tensor`
- **Fix `type(self)` → `Tensor`** in all shape-manipulation methods (`view`, `squeeze`,
  `flatten`, `cpu`, `contiguous`) so subclasses like `Parameter` don't leak through ops
- Introduce `GradFn` base class in `trainadillo/_autograd.py` with `backward()` abstract method
- Implement `Tensor.backward()`: topological sort of the graph, gradient accumulation
- Implement `no_grad()` context manager: suppresses graph construction when active
- Implement `Tensor.detach()`: returns a tensor sharing data but outside the graph

## Capabilities

### New Capabilities

- `autograd-engine`: The core computation graph — `GradFn` base class, `Tensor` autograd
  fields, `backward()`, `no_grad()`, `detach()`. This is the foundation every
  differentiable op (T10 onward) builds on.

### Modified Capabilities

- `tensor-foundations`: Tensor gains three new fields (`grad`, `requires_grad`, `grad_fn`)
  and one derived property (`is_leaf`). The `type(self)` fix changes the return type of
  shape-manipulation methods.

## Non-goals

- Differentiable arithmetic ops (those come in T10; T8 only provides the plumbing)
- Gradient checking utility (T9)
- Any op-specific backward implementations beyond a minimal smoke-test mul

## Impact

- `src/trainadillo/_tensor.py`: add autograd fields; fix `type(self)` in five methods
- `src/trainadillo/_autograd.py`: new file — `GradFn`, graph traversal, `no_grad`
- `src/trainadillo/__init__.py`: export `no_grad`
- Tests: `tests/trainadillo/test_autograd.py` (new)
- No changes to demoodle — trainadillo is self-contained
