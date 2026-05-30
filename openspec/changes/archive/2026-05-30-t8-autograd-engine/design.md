## Context

`Tensor` currently wraps numpy arrays with no memory of how values were produced.
Every operation returns a fresh `Tensor` with no link back to its inputs. Training
is impossible — there is no way to compute gradients.

T8 adds the autograd engine: a computation graph built implicitly during the forward
pass and walked in reverse during `backward()`. Every subsequent milestone (T10
differentiable ops, T11 indexing, T15 cross-entropy, T16 Adam) depends on this
plumbing being correct.

The engine mirrors PyTorch's design: `GradFn` nodes connected by tensor references,
topological sort for the backward walk, gradient accumulation into leaf `.grad`
fields. Trainadillo has no C extensions, no tape, no CUDA — just Python objects
and numpy arrays.

## Goals / Non-Goals

**Goals:**
- Tensor gains `grad`, `requires_grad`, `grad_fn`, `is_leaf` — matching PyTorch's
  public API exactly (no underscore prefixes)
- `GradFn` base class with `backward(grad_output)` abstract method
- `Tensor.backward()` walks the graph via reverse topological sort and accumulates
  gradients into leaf tensors
- `no_grad()` context manager suppresses graph construction
- `Tensor.detach()` returns a tensor outside the graph
- Fix `type(self)` → `Tensor` in five shape-manipulation methods before autograd
  fields exist on the class, so `Parameter` subclass (T12) behaves correctly

**Non-Goals:**
- Differentiable implementations of any ops (those land in T10, T11, T15)
- Gradient checking utility (T9)
- Version counters or in-place mutation guards (not needed — trainadillo has no
  in-place arithmetic ops)
- Thread safety beyond what Python's GIL provides

## Decisions

### D1: All autograd fields are public (no underscores)

`grad`, `requires_grad`, `grad_fn`, `is_leaf` — all public, matching PyTorch's API.
PyTorch exposes all four as public attributes. Adding underscores would mean
trainadillo diverges from PyTorch in a way that breaks the educational goal.

**Alternative considered**: `_grad_fn`, `_is_leaf` as private. Rejected — if the
plan says PyTorch uses these names, reserving underscores for other internals is
cleaner and more teachable.

### D2: `is_leaf` is a derived property, not a stored field

`is_leaf ≡ grad_fn is None`. A tensor is a leaf if and only if nothing in the
autograd system created it. Storing this separately creates a field that can go
out of sync. A `@property` returning `self.grad_fn is None` eliminates the
redundancy with zero cost.

**Implication**: leaf status cannot be set directly — it is a consequence of how
the tensor was created. This matches PyTorch exactly.

### D3: GradFns save Tensor references, not numpy copies

Each `GradFn` holds:
- `inputs: list[Tensor]` — the tensors whose `.grad` will receive contributions
- the forward-pass values it needs as attributes (e.g. `self.a_data = a.data`)

Values are referenced, not copied. This is safe because:
1. Trainadillo has no in-place arithmetic (no `__iadd__`, no `masked_fill_`)
2. The optimizer's `param._data -= update` runs *after* `backward()` completes;
   by the time data is mutated, the GradFn is no longer needed

**Alternative considered**: `.copy()` every saved array. Rejected — the embedding
weight in the bigram model is `(V, V)`; copying it on every forward pass is pure
waste with no safety benefit given the constraints above.

### D4: `type(self)` → `Tensor` in shape-manipulation methods

`view()`, `squeeze()`, `flatten()`, `cpu()`, `contiguous()` currently use
`type(self)(...)`. When `Parameter` subclasses `Tensor` in T12, a reshaped
parameter would return a `Parameter` — which breaks Module's registration logic.

The fix is mechanical: replace `type(self)(...)` with `Tensor(...)` in these five
methods. Shape manipulation produces tensors, not parameters.

**Applied before autograd fields are added** so the fix is isolated and testable
on its own.

### D5: `no_grad()` uses a private state class, matching `_rng.py`

A private `_AutogradState` class holds `grad_enabled: bool = True` as an attribute.
A module-level `_state = _AutogradState()` instance is the single source of truth.
`no_grad.__enter__` and `__exit__` mutate `_state.grad_enabled` — no `global`
needed, and the pattern is already established in `_rng.py`.

```python
class _AutogradState:
    grad_enabled: bool = True

_state = _AutogradState()
```

Trainadillo is single-threaded (no GPU, no DataLoader workers), so `threading.local`
adds complexity with no benefit. If thread safety is ever needed, this is a
one-line swap.

### D6: The backward walker accumulates into `.grad`, creates it on first touch

When `backward()` reaches a leaf tensor, it does:
```python
if tensor.grad is None:
    tensor.grad = Tensor(np.zeros_like(tensor.data))
tensor.grad = tensor.grad + Tensor(grad_contribution)
```

`grad` starts as `None` to distinguish "no gradient computed" from "zero gradient."
This matches PyTorch: `x.grad` is `None` until backward populates it.

## Risks / Trade-offs

**Diamond graphs and gradient accumulation** → The topological sort must process
each GradFn exactly once. Gradients from multiple downstream ops accumulate into
the same buffer before being passed to the shared upstream GradFn. Implement with
a visited set and an `accumulated_grad` dict keyed by GradFn identity.

**Memory: the graph stays alive until GC** → The entire forward-pass graph is
reachable from the output tensor's `grad_fn` chain. For the bigram model this is
trivial, but a 200-step training loop in T19 creates 200 graphs. Python's GC
reclaims them when the loss tensor goes out of scope between steps. No explicit
`zero_grad`-on-graph needed, but worth documenting.

**Non-leaf gradients are discarded** → Only leaf tensors keep `.grad`. Non-leaf
intermediate gradients are transient (passed through the backward walk and then
dropped). This matches PyTorch default behaviour. Diverges from PyTorch only in
that PyTorch raises a warning if you access `.grad` on a non-leaf — trainadillo
can skip the warning for now.

## Open Questions

None — all design decisions resolved during exploration.
