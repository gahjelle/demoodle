# T8 — Autograd Engine

## The concept: recording history to undo it

NumPy knows nothing about where an array's values came from. `c = a + b` gives you `c`, but `c` has no memory of `a` or `b` — it's just numbers. Training a neural network requires the opposite: you need to know not just the output, but the full chain of operations that produced it, so you can differentiate through each one in reverse.

Autograd is the mechanism that provides this memory. During the **forward pass**, every differentiable operation creates a node in an implicit computation graph that records what it did. During the **backward pass**, `loss.backward()` walks that graph in reverse, applying the chain rule at each node to accumulate gradients into the leaf tensors (the model's parameters).

T8 builds the plumbing. No differentiable ops yet — those come in T10–T15. T8 is the engine those ops will attach to.

## The three new structures

### 1. `GradFn` — one node in the computation graph

```
┌──────────────────────────────────────────────────────────┐
│  GradFn (abstract)                                       │
│                                                          │
│  inputs: list[Tensor]    ← what was fed into this op    │
│  backward(grad_output)   ← how to differentiate it      │
│    → list[(Tensor, np.ndarray)]                          │
│      each input paired with its gradient contribution    │
└──────────────────────────────────────────────────────────┘
```

Each concrete subclass captures the forward-pass values it needs for its backward computation. For `Mul(a, b)`, backward is `d/da = grad * b` and `d/db = grad * a` — so `MulGradFn` saves `a.data` and `b.data` at construction time.

The return value is a list of `(tensor, gradient)` pairs, one per input. The backward walker routes each pair to the right accumulation target.

### 2. Autograd fields on `Tensor`

```
grad: Tensor | None          ← accumulated gradient (leaves only)
requires_grad: bool          ← should this tensor track gradients?
grad_fn: GradFn | None       ← the op that created this tensor
is_leaf: property            ← True iff grad_fn is None
```

All four are public, matching PyTorch's API exactly. `is_leaf` is derived — it's not stored, just `return self.grad_fn is None`. A tensor is a leaf if and only if nothing in the autograd system created it.

**Leaf tensors** are user-created (parameters, inputs). `grad_fn=None`, `is_leaf=True`. Their `.grad` accumulates the final gradient.

**Non-leaf tensors** are operation outputs. `grad_fn=<SomeGradFn>`, `is_leaf=False`. Their gradients are transient — passed through the backward walk and discarded.

### 3. The backward walk

`Tensor.backward()` has three phases:

```
Phase 1 — Topological sort
  DFS post-order from self.grad_fn collects all nodes.
  Post-order means a node is appended after all its upstream
  dependencies → reversing gives the correct processing order.

Phase 2 — Seed
  accumulated[id(self.grad_fn)] = 1.0
  (d_loss/d_loss = 1)

Phase 3 — Walk in reverse topological order
  For each GradFn node:
    grad = accumulated[id(node)]
    for (tensor, contrib) in node.backward(grad):
      if tensor.is_leaf and tensor.requires_grad:
        tensor.grad += contrib          ← accumulate
      else:
        accumulated[id(tensor.grad_fn)] += contrib   ← pass upstream
```

The accumulated dict keyed by `id(grad_fn)` is what makes diamond graphs correct.

## Why the topological sort matters

Consider a tensor `x` used by two downstream operations:

```
x ──→ a ──┐
          ├──→ loss
x ──→ b ──┘
```

When `loss.backward()` runs, it must process `loss`'s node before `a`'s and `b`'s nodes. Otherwise, when it gets to `a_grad_fn`, the gradient from `loss` hasn't been computed yet.

Reverse post-order DFS guarantees this. And when both `a_grad_fn` and `b_grad_fn` eventually contribute to `x.grad`, they accumulate (add) rather than overwrite — that's the whole point of the `accumulated` dict for non-leaves and `tensor.grad += contrib` for leaves.

## `no_grad` — turning the graph off

During inference or optimizer steps, you don't want the graph:

```python
with no_grad():
    output = model(x)   # no GradFn nodes created
```

Implementation: a `_AutogradState` class holds `grad_enabled: bool = True`. A module-level `_state = _AutogradState()` is the single source of truth. `no_grad.__enter__` saves and clears it; `__exit__` restores. No `global` needed — attribute mutation on `_state` is sufficient. This mirrors the `_RngState` pattern in `_rng.py`.

Differentiable ops (from T10 onward) check `grad_enabled()` before building a `GradFn`. When `False`, they return plain tensors.

## `detach()` — escaping the graph

`tensor.detach()` returns a new `Tensor` sharing the same numpy array but with `grad_fn=None` and `requires_grad=False`. The data is shared (not copied) — mutations to the detached tensor's values reflect in the original.

Use cases: passing a tensor as data input to a new computation that shouldn't backprop into the original graph; extracting a value for logging without creating a graph reference.

## The `type(self)` fix

T8 also fixes a pre-existing issue in `_tensor.py`. Five shape-manipulation methods (`view`, `squeeze`, `flatten`, `cpu`, `contiguous`) used `type(self)(...)` to construct their return value. When `Parameter` subclasses `Tensor` in T12, calling `view()` on a Parameter would return another Parameter — wrong, since a reshaped view of a weight matrix is not a registered module parameter.

The fix is to use `Tensor(...)` directly. Shape operations always produce Tensors, not subclasses. This is the same invariant PyTorch enforces.

## How real PyTorch does it

The concepts are identical; the implementation layers differ:

| Trainadillo | PyTorch |
|---|---|
| Python `GradFn` objects, `backward()` is a Python method | C++ `Node` objects with `operator()`, called from C++ backward engine |
| Module-level `_AutogradState` flag | Thread-local `GradMode` in C++ (`at::GradMode`) |
| `id(node)` as graph identity | C++ object identity / sequence number |
| No version counters (in-place ops not supported) | Version counters detect if saved tensors were mutated in-place |
| Python DFS for topological sort | C++ queue-based topological execution engine |
| Gradients accumulate with `+=` in Python | Gradients accumulate into pre-allocated buffers in C++ |

The Python-level API — `tensor.grad`, `tensor.grad_fn`, `requires_grad`, `backward()`, `no_grad`, `detach()` — is faithful to PyTorch's.

## What comes next

- **T9 (gradient checking)**: a `gradcheck()` utility that validates backward implementations by comparing analytic gradients to finite differences. Every new op from T10 onward should pass `gradcheck`.
- **T10 (differentiable arithmetic)**: upgrades `+`, `-`, `*`, `/`, `@` to create `GradFn` nodes. Uses `grad_enabled()` and the `inputs` list pattern established in T8.
- **T11 (differentiable indexing)**: `weight[indices]` creates an embedding-lookup `GradFn`. The core backward is `np.add.at(weight_grad, indices, grad_output)`.
- **T15 (cross_entropy)**: the loss function. Its `GradFn` backward is `softmax(logits) - one_hot(targets)`. Everything in the bigram training loop flows through this.
