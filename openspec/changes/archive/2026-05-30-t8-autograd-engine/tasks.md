## 1. Fix Tensor.type(self) → Tensor

- [x] 1.1 In `src/trainadillo/_tensor.py`, replace `type(self)(...)` with `Tensor(...)` in `view()`, `squeeze()`, `flatten()`, `cpu()`, and `contiguous()`
- [x] 1.2 Add tests to `tests/trainadillo/test_tensor.py` confirming that calling `view()`, `squeeze()`, `flatten()` on a `Tensor` subclass returns a plain `Tensor` instance (not the subclass)
- [x] 1.3 Run `uv run pytest tests/trainadillo/test_tensor.py` — all existing tests must still pass

## 2. GradFn base class and `_autograd.py`

- [x] 2.1 Create `src/trainadillo/_autograd.py` with `GradFn` abstract base class holding `inputs: list[Tensor]` and abstract `backward(grad: np.ndarray) -> list[tuple[Tensor, np.ndarray]]`
- [x] 2.2 Add a concrete `_MulGradFn` (package-private) as the minimal smoke-test GradFn: captures `a.data` and `b.data`, implements backward for element-wise multiply
- [x] 2.3 Add `_AutogradState` class with `grad_enabled: bool = True` and a module-level `_state = _AutogradState()` instance, matching the `_RngState` pattern in `_rng.py`
- [x] 2.4 Add `no_grad` context manager: `__enter__` saves and clears `_state.grad_enabled`, `__exit__` restores it
- [x] 2.5 Add `grad_enabled() -> bool` helper returning `_state.grad_enabled` (used by ops in T10 to check before building graph)

## 3. Autograd fields on Tensor

- [x] 3.1 Add `grad: Tensor | None = None`, `requires_grad: bool = False`, `grad_fn: GradFn | None = None` to `Tensor.__init__` in `src/trainadillo/_tensor.py` (use `TYPE_CHECKING` import of `GradFn` to avoid circular import)
- [x] 3.2 Add `is_leaf` as a `@property` returning `self.grad_fn is None`
- [x] 3.3 Update `Tensor.__init__` signature to accept `requires_grad: bool = False` as a keyword argument
- [x] 3.4 Add tests: default state (`grad=None`, `requires_grad=False`, `grad_fn=None`, `is_leaf=True`); constructing with `requires_grad=True`; `is_leaf=False` when `grad_fn` is set manually

## 4. `Tensor.backward()` and `Tensor.detach()`

- [x] 4.1 Implement `Tensor.backward()` in `_tensor.py`: assert scalar shape; run reverse topological sort via DFS from `self.grad_fn`; walk nodes accumulating gradients; populate `.grad` on leaves (create zeros on first touch, then accumulate)
- [x] 4.2 Implement `Tensor.detach()`: return `Tensor(self.data)` with `requires_grad=False` and `grad_fn=None`, sharing the same underlying numpy array
- [x] 4.3 Add tests in `tests/trainadillo/test_autograd.py`:
  - Simple scalar: `x = Tensor(3.0, requires_grad=True)`, `y = x * 2` via `_MulGradFn`, `y.backward()`, assert `x.grad.item() == 2.0`
  - Diamond graph: `a = f(x)`, `b = g(x)`, `loss = a + b` (wire manually with GradFns), assert `x.grad` is the sum
  - `backward()` on non-scalar raises
  - `no_grad` suppresses `grad_fn`
  - `detach()` shares data, has no `grad_fn`

## 5. Package wiring

- [x] 5.1 Export `no_grad` and `GradFn` from `src/trainadillo/__init__.py`
- [x] 5.2 Export `grad_enabled` (as a private helper — no need to expose publicly, but confirm it works internally)

## 6. Verification

- [x] 6.1 `uv run ruff format src/trainadillo/ tests/trainadillo/`
- [x] 6.2 `uv run ruff check src/trainadillo/ tests/trainadillo/`
- [x] 6.3 `uv run ty check src/ tests/`
- [x] 6.4 `uv run pytest tests/trainadillo/` — full trainadillo test suite passes

## 7. Docs and project tracking

- [x] 7.1 Write `docs/trainadillo/008-autograd-engine.md` explaining the computation graph, GradFn design, topological sort, and how `no_grad` works — connect to PyTorch's C++ equivalent
- [x] 7.2 Mark T8 done in `PLANS_TRAINADILLO.md` with ✅
- [x] 7.3 Review `CONTEXT.md` — add or update glossary terms for `GradFn`, `grad_fn`, `requires_grad`, `backward pass`, `leaf tensor`
