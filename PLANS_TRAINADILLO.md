# Trainadillo — Work Items (Build Order)

> Ordered backlog for building `trainadillo`, a minimal PyTorch clone
> on NumPy. Each item is self-contained enough to drop into a prompt. Each lists
> **goal · build · done-when · depends-on**. Implement top to bottom.
>
> Trainadillo knows nothing about demoodle — imports go one way only. All
> trainadillo code lives under `src/trainadillo/` but has zero imports
> from `demoodle.*`. Tests live under `tests/trainadillo/`.
>
> The general autograd strategy: every differentiable op creates a `GradFn` node
> storing a backward closure. `Tensor.backward()` walks the graph in reverse
> topological order, calling each closure and accumulating gradients into leaf
> tensors. This mirrors PyTorch's design and makes adding new ops incremental.

---

## Educational Intent

**Trainadillo is a learning project.** The goal is to understand how PyTorch works
under the hood by building its core ideas from scratch on NumPy.

When implementing any T-item, take the time to:
- Explain the **why** behind each design decision, not just the what
- Connect the concept back to how real PyTorch implements it (and where trainadillo
  simplifies — no GPU, no C extensions, no autograd tape in C++)
- Highlight non-obvious invariants or subtleties (e.g. why `__bool__` is absent,
  why integer indexing returns a 0-D Tensor rather than a scalar)
- After implementing, write an explanatory note to
  `docs/trainadillo/<00N>-<short-name>.md` that captures the concepts introduced

---

## Milestone T0 — Tensor Foundations

### ✅ T1. Tensor class & dtype constants

- **Goal:** the core data type wrapping `numpy.ndarray`, without autograd.
- **Build:** `trainadillo/_tensor.py` — a `Tensor` class holding `_data: np.ndarray`.
  Properties: `shape`, `ndim`, `dtype`, `data` (returns `_data`). Methods:
  `item()` (extract scalar), `tolist()` (convert to Python list), `size(dim=None)`
  (return shape or shape[dim]), `cpu()` (no-op, return self), `contiguous()` (no-op),
  `view(*shape_or_dtype)` (delegate to `np.ndarray.reshape` for shape args or
  `np.ndarray.view` for dtype args — needed by persistence.py's
  `.view(torch.uint8)` pattern), `squeeze(dim=None)`, `flatten()`.
  `__getitem__` for indexing/slicing (return a new Tensor wrapping the numpy result;
  integer indexing on a 1-D tensor returns a 0-D Tensor, not a Python scalar).
  Arithmetic dunder methods (`__add__`, `__sub__`, `__mul__`, `__truediv__`,
  `__neg__`) — for now these are non-differentiable, just wrapping numpy.
  Comparison dunder methods (`__gt__`, `__ge__`, `__lt__`, `__le__`, `__eq__`) —
  return boolean Tensors. **Do not implement `__bool__`** — boolean Tensors must
  be usable as masks (e.g. `(cumulative - sorted_probs) > top_p` produces a
  boolean Tensor passed to `masked_fill`), but converting a multi-element Tensor
  to a Python bool should raise an error (matching PyTorch behavior).
  `__repr__` and `__len__`.
  Also define module-level dtype constants: `long = np.int64`, `uint8 = np.uint8`,
  `float32 = np.float32` (the default).
  **Do not** add autograd fields yet — those come in G8 when the autograd engine
  is built. Keep G1 focused on the numpy wrapper.
- **Done when:** `Tensor` wraps arrays; `item()`, `tolist()`, `shape`, `view`,
  indexing, arithmetic all work; `tensor([1, 2, 3], dtype=long)` produces an int64
  Tensor; a test confirms `.view(uint8)` on a float32 Tensor returns raw bytes;
  comparison ops produce boolean Tensors that work as masks.
- **Depends on:** —

### ✅ T2. Creation functions

- **Goal:** factory functions matching the `torch.*` creation API.
- **Build:** `trainadillo/_creation.py` — functions that return `Tensor` objects:
  `tensor(data, *, dtype=None)` (convert Python data + optional dtype into a
  Tensor), `zeros(*shape)`, `ones(*shape)`, `zeros_like(t)`, `full_like(t, value)`,
  `arange(n)`, `stack(tensors, dim=0)`, `equal(a, b)` (element-wise comparison,
  return Python bool — True iff all elements match, like `torch.equal`).
  Each wraps the corresponding `np.*` call and returns a `Tensor`.
  **Note on tensor arithmetic with Python ints:** the codebase does
  `tokens[offsets + context_len]` where `offsets` is a Tensor and `context_len`
  is a Python int. This works naturally since G1's `__add__` wraps numpy, which
  handles int+array. Verify this case explicitly in tests.
- **Done when:** all functions return `Tensor`; `stack` combines a list of 1-D
  Tensors into a 2-D Tensor; `equal` returns True for identical tensors and False
  otherwise; `tensor(data) + python_int` works.
- **Depends on:** G1

### T3. Generator & RNG

- **Goal:** reproducible random number generation matching the `torch.Generator`
  interface.
- **Build:** `trainadillo/_rng.py` — a `Generator` class wrapping
  `numpy.random.Generator` (backed by `PCG64`). Method: `manual_seed(seed)` —
  (re)initializes the internal numpy generator with the given seed. Also a
  module-level `_default_generator` and a `manual_seed(seed)` function that
  seeds it. The `Generator` must be passable to creation functions in G4.
- **Done when:** two `Generator`s with the same seed produce identical sequences;
  different seeds diverge; `manual_seed` at module level works.
- **Depends on:** —

### T4. Random creation functions

- **Goal:** random tensor factories that accept an optional `generator` parameter.
- **Build:** add to `trainadillo/_creation.py` (or a separate `_random.py`):
  `rand(*shape, generator=None)` (uniform [0,1)),
  `randint(low, high, size, *, generator=None)` (random integers, note: PyTorch's
  `randint` takes `size` as a tuple, not `*args`).
  When `generator` is None, use the module-level default generator.
- **Done when:** `randint(0, 10, (5,), generator=g)` returns a shape-(5,) int64
  Tensor; same generator seed produces same output; different seeds produce
  different output.
- **Depends on:** G1, G3

---

## Milestone T1 — Tensor Operations (Non-Differentiable)

> These ops are used in sampling (`_sample()` in `bigram.py`) and in data
> preprocessing. They are never in the gradient path, so they do not need
> backward implementations. Build them as pure numpy wrappers.

### T5. Reduction & sorting ops

- **Goal:** `topk`, `sort`, `argmax` on Tensors.
- **Build:** in `trainadillo/_ops.py`:
  `topk(tensor, k, dim=-1)` returns `(values, indices)` as a pair of Tensors
  (use `np.argpartition` + `np.argsort` for efficiency, or just `np.argsort` for
  simplicity since tensors are small).
  `sort(tensor, dim=-1, descending=False)` returns `(sorted_values, indices)`.
  Also add `Tensor.argmax(dim=None)` as a method delegating to `np.argmax`.
- **Done when:** `topk` returns the k largest values and their indices; `sort` with
  `descending=True` works; `argmax` returns the index of the maximum.
- **Depends on:** G1

### T6. Softmax, cumsum, multinomial

- **Goal:** the probability-distribution operations used in sampling.
- **Build:** in `trainadillo/_ops.py`:
  `softmax(tensor, dim=-1)` — the numerically stable version
  (`exp(x - max(x)) / sum(exp(x - max(x)))`). This is the standalone function
  form used during inference — *not* the differentiable version needed for
  training (that comes in G15).
  `cumsum(tensor, dim)` — cumulative sum along a dimension.
  `multinomial(probs, num_samples, *, generator=None)` — sample indices from a
  categorical distribution. Use the numpy generator's `choice` method with the
  probability weights. Return a Tensor of sampled indices.
- **Done when:** `softmax` output sums to 1; `cumsum` matches numpy; `multinomial`
  draws from the distribution (statistical test: over many draws, frequencies
  approximate the input probabilities); the `generator` parameter makes results
  reproducible.
- **Depends on:** G1, G3

### T7. Masking ops & scatter

- **Goal:** `masked_fill` and `scatter_` used in top-k/top-p filtering.
- **Build:** in `trainadillo/_ops.py`:
  `Tensor.masked_fill(mask, value)` — return a new Tensor where positions where
  `mask` is True are replaced with `value`. (Note: PyTorch's `masked_fill` is
  in-place with an underscore variant, but the codebase uses the non-in-place form
  via method syntax: `sorted_logits.masked_fill(to_remove, float("-inf"))`. Support
  both.)
  `Tensor.scatter_(dim, index, src)` — in-place scatter. Writes values from `src`
  into `self` at positions given by `index` along `dim`. This modifies the tensor
  in-place (returns self for chaining). For the dim=0 case used by the codebase,
  the numpy equivalent is fancy indexing: `self._data[index._data] = src._data`.
  For general dims, iterate or use `np.put_along_axis` (but note
  `np.put_along_axis` has slightly different semantics — test carefully).
  The codebase uses: `mask.scatter_(0, top_indices, scaled[top_indices])` and
  `torch.zeros_like(scaled).scatter_(0, sorted_indices, sorted_logits)`.
- **Done when:** `masked_fill` replaces correct positions; `scatter_` places values
  at the right indices; the full `_sample()` function from bigram.py works when
  using trainadillo tensors (test with a known logit vector).
- **Depends on:** G1, G2

---

## Milestone T2 — Autograd Engine

### T8. Computation graph & backward pass

- **Goal:** the autograd core — build a computation graph and walk it backward.
- **Build:** `trainadillo/_autograd.py`:
  A `GradFn` base class (or protocol) with an abstract `backward(grad_output)`
  method that returns a list of `(Tensor, np.ndarray)` pairs (each input tensor
  paired with its gradient contribution).
  Extend the `Tensor` class with autograd fields: `grad: Tensor | None = None`,
  `requires_grad: bool = False`, `_grad_fn: GradFn | None = None`,
  `_is_leaf: bool = True`. These were not in G1 — add them now that the autograd
  engine exists to use them.
  When `requires_grad` is True and `no_grad` mode is off, operations
  record a `_grad_fn` on the output tensor pointing back to the inputs.
  `Tensor.backward()`: assert this is a scalar; set its grad to 1.0; do a reverse
  topological sort of the graph reachable from `_grad_fn`; for each node, call
  `backward(accumulated_grad)` and accumulate results into the leaf tensors' `.grad`
  attributes (create `.grad` as zeros on first touch, then add).
  `no_grad()` context manager: sets a thread-local (or module-level) flag; when
  active, all ops produce tensors with `_grad_fn = None` regardless of input
  `requires_grad`.
  Also add `Tensor.detach()`: returns a new Tensor sharing the same `_data` but
  with `requires_grad=False` and `_grad_fn=None`.
- **Done when:** a minimal test: create a leaf tensor `x` with `requires_grad=True`,
  compute `y = x * 2` (using a simple mul GradFn), call `y.backward()`, and verify
  `x.grad` equals 2. `no_grad` suppresses graph construction. Topological sort
  handles diamond-shaped graphs (a tensor used by two downstream ops — gradients
  accumulate, not overwrite).
- **Depends on:** G1

### T9. Gradient checking utility

- **Goal:** a reusable tool for validating backward implementations.
- **Build:** `trainadillo/_grad_check.py`:
  `gradcheck(fn, inputs, eps=1e-5, atol=1e-4, rtol=1e-3)` — for each input
  with `requires_grad=True`, compute the Jacobian numerically (central finite
  differences: `(f(x+eps) - f(x-eps)) / 2eps`) and analytically (via backward),
  assert they match within tolerance.
  This is a test utility, not part of the public API — but it is essential for
  validating every differentiable op from G10 onward. Build it now so that every
  subsequent "done when" clause can say "gradient check passes" and mean something
  concrete.
- **Done when:** correctly catches an intentionally-wrong backward (e.g. `grad * 3`
  instead of `grad * 2` for a `x * 2` op) and passes a correct one.
- **Depends on:** G8

### T10. Differentiable arithmetic ops

- **Goal:** make `+`, `-`, `*`, `/`, `**`, `@` (matmul) track gradients.
- **Build:** upgrade the arithmetic dunders from G1 to create `GradFn` nodes when
  either operand has `requires_grad=True` (and `no_grad` is off). Each op needs a
  backward closure:
  - **Add:** `d/da = grad`, `d/db = grad` (broadcast-aware: sum over broadcast dims)
  - **Sub:** `d/da = grad`, `d/db = -grad`
  - **Mul:** `d/da = grad * b`, `d/db = grad * a`
  - **TrueDiv:** `d/da = grad / b`, `d/db = -grad * a / b^2`
  - **Neg:** `d/da = -grad`
  - **Matmul (`@`):** `d/da = grad @ b.T`, `d/db = a.T @ grad`
  Also handle scalar-tensor mixed ops (e.g. `logits / temperature` where
  temperature is a Python float). The float doesn't need a gradient — just
  propagate to the tensor operand.
  Broadcasting: when input was broadcast during forward, the backward must sum
  over the broadcast dimensions to match the original shape. Implement a helper
  `_unbroadcast(grad, target_shape)` that does this.
- **Done when:** gradient check (G9) passes for each op with various shapes
  including broadcast cases.
- **Depends on:** G8, G9

### T11. Differentiable indexing

- **Goal:** make `weight[x]` (parameter lookup by integer index) differentiable.
  This is the critical op for the bigram model.
- **Build:** upgrade `Tensor.__getitem__` to create a `GradFn` when the indexed
  tensor has `requires_grad=True`. The backward for integer indexing:
  - Forward: `out = weight[indices]` where `weight` is (V, V) and `indices` is a
    batch of token IDs.
  - Backward: `weight.grad` is (V, V) initialized to zeros; for each index `i` in
    `indices`, add `grad_output[i]` to `weight.grad[indices[i]]`. This is an
    `np.add.at` operation: `np.add.at(weight_grad, indices, grad_output)`.
  Support both single-integer indexing (`weight[3]`) and batch indexing
  (`weight[tensor_of_ints]`). Slice indexing (`tensor[2:5]`) can remain
  non-differentiable for now (not used in training path).
- **Done when:** gradient check (G9) passes: index into a parameter with various
  indices, backward, compare with finite differences. The bigram forward pass
  `self.weight[x]` is differentiable.
- **Depends on:** G8, G9

---

## Milestone T3 — Module System

### T12. Parameter & Module

- **Goal:** `nn.Module` and `nn.Parameter` matching the PyTorch interface used by
  demoodle.
- **Build:** `trainadillo/nn/_parameter.py`: `Parameter` is a `Tensor` (or thin
  wrapper) with `requires_grad=True` by default. It must be recognizable by
  `Module.__setattr__` for automatic registration.
  `trainadillo/nn/_module.py`: `Module` base class. Key mechanics:
  - `__init__` must initialize `_parameters` and `_modules` dicts **before** the
    `__setattr__` intercept is active. Use `object.__setattr__(self, '_parameters', {})`
    and `object.__setattr__(self, '_modules', {})` in `Module.__init__()` to
    avoid triggering the intercept during setup. This is a subtle ordering issue —
    PyTorch solves it the same way.
  - `__setattr__` intercepts assignments: if the value is a `Parameter`, register
    it in `_parameters` dict; if it's a `Module`, register in `_modules` dict;
    otherwise, normal attribute.
  - `parameters()` yields all `Parameter`s recursively (own + children).
  - `named_parameters(prefix="")` yields `(dotted_name, Parameter)` pairs.
  - `state_dict()` returns `{name: Tensor}` dict of all parameters.
  - `load_state_dict(d)` loads parameters from a dict.
  - `__call__(*args, **kwargs)` delegates to `self.forward(...)`.
  - `forward(...)` — abstract, must be overridden.
  - `cpu()` — no-op, returns self.
  - `train(mode=True)` / `eval()` — sets `self.training` flag (for dropout later).
- **Done when:** a simple module with two Parameters can be constructed; `parameters()`
  yields both; `state_dict()` round-trips through `load_state_dict()`;
  `named_parameters()` produces correct dotted paths for nested modules; the
  existing `BigramModel` class definition works with `from trainadillo import nn`.
- **Depends on:** G1, G8

### T13. nn.Linear

- **Goal:** dense layer used extensively in tests and needed for MLP/transformer.
- **Build:** `trainadillo/nn/_linear.py`: `Linear(in_features, out_features, bias=True)`.
  Holds `weight` Parameter of shape `(out_features, in_features)` and optional
  `bias` Parameter of shape `(out_features,)`. Forward: `x @ weight.T + bias`.
  Initialization: Kaiming uniform (same as PyTorch default) — or simpler: uniform
  in `[-1/sqrt(in_features), 1/sqrt(in_features)]`.
  The forward pass uses the differentiable matmul and add ops from G10, so backward
  comes for free.
- **Done when:** `Linear(4, 3)(x)` produces output of shape `(*, 3)`;
  `model.parameters()` includes weight and bias; backward through Linear works
  (gradient check). Tests that create `Policy(model=nn.Linear(4, 4))` work.
- **Depends on:** G10, G12

---

## Milestone T4 — Loss, Optimizer, Init

### T14. nn.init (parameter initialization)

- **Goal:** parameter initialization functions.
- **Build:** `trainadillo/nn/init.py`:
  `normal_(tensor, mean=0.0, std=1.0, *, generator=None)` — fills tensor in-place
  with values from a normal distribution using the given generator. Returns the
  tensor. This is used by `BigramArchitecture.init_state()` to initialize the
  weight matrix.
- **Done when:** `nn.init.normal_` fills a tensor with seeded random values; same
  generator seed produces same initialization; the tensor is modified in-place.
- **Depends on:** G1, G3

### T15. F.cross_entropy & differentiable softmax

- **Goal:** the training loss function — the hardest autograd op in the bigram path.
- **Build:** `trainadillo/nn/functional.py`:
  `cross_entropy(input, target)` — takes logits of shape `(N, C)` or `(C,)` and
  integer targets of shape `(N,)` or scalar. Computes `log_softmax(input)` then
  negative log-likelihood at the target indices. Returns a scalar loss Tensor.
  **This must be differentiable.** The backward:
  `d_loss/d_logits = softmax(logits) - one_hot(targets)` (divided by batch size
  for mean reduction). Implement as a single fused `GradFn` for numerical stability
  (log-sum-exp trick) rather than composing softmax, log, nll.
  Also add `softmax(input, dim)` as a differentiable function in functional.py
  (will be needed for attention in G28). Backward of softmax:
  `s = softmax(input)`, `d_input = s * (grad - sum(grad * s, dim, keepdim=True))`.
- **Done when:** `F.cross_entropy(logits, targets)` returns a scalar; calling
  `.backward()` on it populates gradients on the logits tensor; gradient check
  (G9) passes for both `cross_entropy` and `softmax`.
- **Depends on:** G8, G9, G11

### T16. Adam optimizer

- **Goal:** the optimizer used in the training loop.
- **Build:** `trainadillo/optim/_adam.py`: `Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)`.
  `params` is an iterable of Parameters (from `model.parameters()`).
  Methods:
  - `zero_grad()` — set `.grad` to None (or zeros) on all parameters.
  - `step()` — update parameters using the Adam rule:
    maintain per-parameter first moment `m` and second moment `v`;
    `m = beta1 * m + (1 - beta1) * grad`;
    `v = beta2 * v + (1 - beta2) * grad^2`;
    bias-corrected: `m_hat = m / (1 - beta1^t)`, `v_hat = v / (1 - beta2^t)`;
    `param -= lr * m_hat / (sqrt(v_hat) + eps)`.
    Increment step counter `t` on each call.
    Modify `param._data` in-place (do not create new Tensor objects — the
    Module's references must remain valid).
- **Done when:** a simple optimization loop on a known convex loss (e.g.
  `(x - 3)^2`) converges `x` toward 3; the optimizer's `zero_grad()` clears
  gradients; the training loop pattern
  `optimizer.zero_grad(); loss.backward(); optimizer.step()` works end-to-end.
- **Depends on:** G12 (Parameter), G10 (arithmetic for test loss)

---

## Milestone T5 — Serialization & Wiring

### T17. Serialization (save / load)

- **Goal:** persist and reload trained models.
- **Build:** `trainadillo/_serialization.py`:
  `save(obj, path)` — serialize `obj` to `path` using `pickle`. Add
  `Tensor.__reduce__` (or `__getstate__`/`__setstate__`) so that only the numpy
  array and `requires_grad` flag are pickled — not the computation graph (`_grad_fn`,
  backward closures, etc.).
  `load(path, *, weights_only=False)` — deserialize from `path` using `pickle`.
  Accept the `weights_only` parameter for API compatibility but do not enforce it
  (it is a PyTorch security feature for untrusted data).
  **Note on format:** this uses Python's pickle, not PyTorch's internal format.
  Files saved by trainadillo cannot be loaded by `torch.load` and vice versa. The
  demoodle artifact cache (persistence.py) will need its existing cached `.pt`
  files cleared after migration in G20. Consider whether to keep the `.pt`
  extension (less churn in persistence.py) or change it (clearer that the format
  differs) — decide during G20 implementation.
- **Done when:** round-trip a dict of Tensors through save/load; round-trip a Module
  subclass through save/load; the computation graph is not serialized (loaded tensors
  are leaves with `_grad_fn=None`).
- **Depends on:** G1, G12

### T18. Package wiring (__init__.py files)

- **Goal:** make `import trainadillo as torch` work with the full public API.
- **Build:** write `__init__.py` files that re-export everything at the right level:
  `trainadillo/__init__.py` exports: `Tensor`, creation functions (`zeros`, `ones`,
  `tensor`, `arange`, `randint`, `stack`, `rand`, `full_like`, `zeros_like`,
  `equal`), ops (`topk`, `sort`, `softmax`, `cumsum`, `multinomial`), dtype
  constants (`long`, `uint8`, `float32`), `Generator`, `manual_seed`, `no_grad`,
  `save`, `load`, and the `nn` and `optim` subpackages.
  `trainadillo/nn/__init__.py` exports: `Module`, `Parameter`, `Linear`, and the
  `functional` and `init` submodules (so `nn.functional` and `nn.init` work as
  attribute access).
  `trainadillo/optim/__init__.py` exports: `Adam`.
  Verify that these import patterns all work:
  - `import trainadillo as torch; torch.zeros(3)`
  - `from trainadillo import nn; nn.Module`
  - `import trainadillo as torch; torch.nn.functional.cross_entropy`
  - `from trainadillo import nn; import trainadillo.nn.functional as F`
- **Done when:** all four import patterns resolve correctly; no circular imports;
  `dir(trainadillo)` shows the expected public API.
- **Depends on:** G1–G17

---

## Milestone T6 — Integration

### T19. Bigram integration test

- **Goal:** prove trainadillo can train the bigram model end-to-end without any changes
  to model or training code (only import lines change).
- **Build:** a standalone integration test under `tests/trainadillo/` that:
  1. Copies the `BigramModel` class and `_sample()` function verbatim (changing
     only `import torch` to `import trainadillo as torch`).
  2. Creates a small synthetic corpus (e.g. 100 repeated "abc\n" strings).
  3. Encodes it to a token tensor.
  4. Runs the training loop (Adam, cross_entropy, backward, step) for 200 steps.
  5. Asserts loss decreased by at least 50%.
  6. Runs `_sample()` and asserts it returns a valid token id.
  7. Tests save/load round-trip of the trained model.
  8. Tests that the full `_sample()` code path works (temperature, top-k, top-p).
  Do **not** import from demoodle in this test — keep trainadillo self-contained.
  Copy the code that is needed.
- **Done when:** the integration test passes; loss decreases; sampling works;
  save/load round-trips. This proves trainadillo is a sufficient PyTorch replacement
  for the current codebase.
- **Depends on:** G18

### T20. Migrate demoodle imports

- **Goal:** remove PyTorch as a dependency; use trainadillo everywhere.
- **Build:** in every file that currently has `import torch` or `from torch import`:
  replace with the trainadillo equivalent. Files to update:
  - `src/demoodle/core/types.py` — type annotations use `trainadillo.Tensor`,
    `trainadillo.nn.Module`
  - `src/demoodle/core/rng.py` — `trainadillo.Generator` instead of `torch.Generator`
  - `src/demoodle/architectures/bigram.py` — model + sampling
  - `src/demoodle/training/stages.py` — training loop
  - `src/demoodle/shell/persistence.py` — save/load/hashing
  - `src/demoodle/data/stages.py` — dataset creation
  - `src/demoodle/frontends/cli.py` — tensor creation in generation
  - All test files under `tests/`
  Remove `torch` from `pyproject.toml` dependencies. Ensure `numpy` is listed
  (it may already be a transitive dependency).
  **Cache invalidation:** delete all cached `.pt` files under the cache directory.
  The serialization format changes from torch's pickle protocol to plain pickle —
  old caches are unloadable. Update ADR-0004 to reference `trainadillo.nn.Module`
  instead of `torch.nn.Module`. Update CONTEXT.md glossary entries for Policy,
  Artifact Cache, and Output to reference trainadillo types.
- **Done when:** `uv run pytest` passes; `uv run demoodle train` trains and shows
  decreasing loss; `uv run demoodle call` generates plausible names; `torch` is not
  importable (not installed) and everything still works; `uv run ruff check .`
  passes.
- **Depends on:** G19

---

## Milestone T7 — MLP Architecture Support

> These items add the ops needed for W13 (MLP architecture) from PLANS.md.
> Build them when W13 is being implemented, or ahead of time for practice.

### T21. nn.Embedding

- **Goal:** learnable embedding lookup table.
- **Build:** `trainadillo/nn/_embedding.py`: `Embedding(num_embeddings, embedding_dim)`.
  Holds a `weight` Parameter of shape `(num_embeddings, embedding_dim)`.
  Forward: `self.weight[input]` — indexes into the weight matrix (uses the
  differentiable indexing from G11, which already handles batched integer indices).
  Initialization: normal distribution (mean=0, std=1).
  Add to `nn/__init__.py` exports.
- **Done when:** `Embedding(10, 4)(tensor([2, 5, 7]))` returns shape `(3, 4)`;
  backward populates gradients on `weight` only at the indexed rows; gradient
  check passes.
- **Depends on:** G11, G12

### T22. Activation functions (ReLU, GELU)

- **Goal:** nonlinearities for MLP hidden layers.
- **Build:** in `trainadillo/nn/functional.py`:
  `relu(input)` — `max(0, input)`. Backward: `grad * (input > 0)`.
  `gelu(input)` — the Gaussian Error Linear Unit. Use the approximation:
  `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
  Backward: compute analytically or via the exact formula derivative.
  Both must create `GradFn` nodes for autograd.
  Optionally add `nn.ReLU()` and `nn.GELU()` as Module wrappers that call the
  functional versions in their `forward()`.
- **Done when:** `relu` zeros out negatives, passes positives; `gelu` approximates
  the Gaussian CDF-weighted identity; gradient check passes for both.
- **Depends on:** G8, G9

### T23. Sum & mean reductions

- **Goal:** differentiable reduction ops needed as building blocks.
- **Build:** add `Tensor.sum(dim=None, keepdim=False)` and
  `Tensor.mean(dim=None, keepdim=False)` as differentiable ops.
  Sum backward: broadcast `grad_output` back to input shape.
  Mean backward: `grad_output / n` broadcast back to input shape.
  These are needed internally by loss functions and will be used by layer norm.
- **Done when:** `x.sum()` and `x.mean()` produce correct values; backward works;
  gradient check passes for various shapes and dim arguments.
- **Depends on:** G8, G9

---

## Milestone T8 — Transformer Architecture Support

> These items add the ops needed for W16 (tiny transformer) from PLANS.md.
> Build them when W16 is being implemented, or ahead of time for practice.

### T24. Transpose & reshape in autograd

- **Goal:** shape manipulation that preserves gradient flow.
- **Build:** make `Tensor.view(*shape)`, `Tensor.reshape(*shape)`,
  `Tensor.transpose(dim0, dim1)`, and `Tensor.permute(*dims)` differentiable.
  These don't change the data, just the shape/strides, so backward is just
  reshaping the gradient back to the original shape:
  - view/reshape backward: `grad.reshape(input_shape)`
  - transpose backward: `grad.transpose(dim0, dim1)`
  - permute backward: `grad.permute(inverse_permutation)`
  Also add the `.T` property (transpose for 2-D tensors).
- **Done when:** gradient flows through reshape, matmul, reshape chains correctly;
  gradient check passes.
- **Depends on:** G8, G9

### T25. nn.LayerNorm

- **Goal:** layer normalization for transformer blocks.
- **Build:** `trainadillo/nn/_layernorm.py`:
  `LayerNorm(normalized_shape, eps=1e-5)`. Holds learnable `weight` and `bias`
  Parameters of shape `normalized_shape`.
  Forward: normalize the last `len(normalized_shape)` dimensions to zero mean and
  unit variance, then scale by `weight` and shift by `bias`.
  Backward: the layer norm backward is well-documented but involves several terms.
  Save the normalized input and standard deviation from the forward pass for
  reuse in backward (avoiding recomputation). The gradient w.r.t. input `x`:
  let `x_hat = (x - mean) / std`, then
  `dx = (gamma / std) * (dy - mean(dy) - x_hat * mean(dy * x_hat))`,
  where `dy` is the upstream gradient.
  Implement carefully and validate with gradient checking.
  Add to `nn/__init__.py` exports.
- **Done when:** output has zero mean and unit variance (before affine); gradient
  check passes; matches expected behavior on known inputs.
- **Depends on:** G12, G23

### T26. nn.Dropout

- **Goal:** regularization via random zeroing during training.
- **Build:** `trainadillo/nn/_dropout.py`: `Dropout(p=0.5)`.
  Forward (training mode): generate a random mask where each element is 0 with
  probability `p` and 1 otherwise; multiply input by mask and scale by `1/(1-p)`
  (inverted dropout).
  Forward (eval mode): identity (pass through unchanged).
  Backward: `grad * mask / (1-p)` (same mask as forward).
  Uses `Module.training` flag (set by `.train()` / `.eval()` from G12).
  Add to `nn/__init__.py` exports.
- **Done when:** in training mode, approximately `p` fraction of outputs are zero;
  in eval mode, output equals input; gradient flows through non-zeroed positions;
  gradient check passes.
- **Depends on:** G8, G9, G12

### T27. nn.Sequential

- **Goal:** ordered container of modules.
- **Build:** `trainadillo/nn/_sequential.py`: `Sequential(*modules)`.
  Forward: pass input through each module in order.
  Inherits `parameters()`, `state_dict()`, etc. from `Module` — child modules
  are registered via `__setattr__` with string-integer keys ("0", "1", ...).
  Add to `nn/__init__.py` exports.
- **Done when:** `Sequential(Linear(4, 8), ReLU(), Linear(8, 2))` works; parameters
  from all children are accessible; backward flows through the chain.
- **Depends on:** G12, G13, G22

### T28. Batched matmul & attention primitives

- **Goal:** the matrix operations needed for multi-head self-attention.
- **Build:** ensure matmul (`@` operator from G10) handles batched inputs:
  `(B, N, D) @ (B, D, M) -> (B, N, M)`. The backward must handle these shapes
  correctly — for batched matmul, the gradients are:
  `d/da = grad @ b.transpose(-2, -1)`, `d/db = a.transpose(-2, -1) @ grad`,
  applied per-batch.
  Add `Tensor.bmm(other)` as an alias if needed.
  Add a differentiable `masked_fill` for the causal attention mask: when used
  on a tensor with `requires_grad=True`, the backward passes gradient through
  non-masked positions and zeros through masked positions. (The non-differentiable
  `masked_fill` from G7 only handles the inference path.)
  These are the primitives. The actual multi-head attention (Q/K/V projections,
  scaled dot-product, concatenation) will be built in the transformer architecture
  code in demoodle, not in trainadillo — trainadillo just provides the building blocks.
- **Done when:** `(B, N, D) @ (B, D, M)` works in forward and backward; gradient
  check passes for batched matmul; masked_fill backward zeros out gradients at
  masked positions.
- **Depends on:** G10, G9, G24

---

## Milestone T9 — RLHF Support

> These items add ops needed for W22–W27 (SFT, DPO, PPO) from PLANS.md.
> Build them when those work items are being implemented.

### T29. Log, exp, sigmoid, and KL divergence

- **Goal:** differentiable transcendental functions needed by DPO/PPO losses.
- **Build:** in `trainadillo/_ops.py` or `nn/functional.py`:
  `Tensor.log()` — natural logarithm. Backward: `grad / input`.
  `Tensor.exp()` — exponential. Backward: `grad * exp(input)`.
  `torch.sigmoid(input)` — `1 / (1 + exp(-x))`. Backward:
  `grad * sigmoid * (1 - sigmoid)`.
  `F.log_softmax(input, dim)` — log of softmax (numerically stable:
  `x - log(sum(exp(x)))`). Backward: `grad - softmax * sum(grad, dim)`.
  `F.kl_div(log_input, target, reduction)` — KL divergence.
  These compose to build DPO loss (`log_sigmoid(beta * (log_ratio_w - log_ratio_l))`)
  and PPO's clipped surrogate objective.
- **Done when:** gradient check passes for each function; `log_softmax` is
  numerically equivalent to `log(softmax(x))` but stable for large values.
- **Depends on:** G8, G9

### T30. MSE loss & clamp

- **Goal:** loss function for reward model training; clamp for PPO.
- **Build:**
  `F.mse_loss(input, target)` — mean squared error. Backward:
  `2 * (input - target) / n`.
  `Tensor.clamp(min=None, max=None)` — clamp values. Backward: gradient passes
  through where value is in range, zero where clamped.
  These are needed for the reward model (W25) and PPO's clipped ratio (W27).
- **Done when:** `mse_loss` returns correct value; `clamp` clips correctly;
  gradient check passes for both.
- **Depends on:** G8, G9
