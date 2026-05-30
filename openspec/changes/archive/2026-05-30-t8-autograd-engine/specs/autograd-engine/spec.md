## ADDED Requirements

### Requirement: `GradFn` is the base class for all backward functions

`GradFn` SHALL be an abstract base class in `trainadillo/_autograd.py`. It SHALL
expose a `backward(grad_output: np.ndarray) -> list[tuple[Tensor, np.ndarray]]`
method. Each tuple in the return value pairs an input tensor with its gradient
contribution. Concrete subclasses SHALL store whatever forward-pass values are
needed to compute the gradient.

#### Scenario: GradFn backward returns one pair per input
- **WHEN** a concrete `GradFn` subclass is called with `backward(grad)`
- **THEN** it returns a list of `(Tensor, np.ndarray)` pairs, one per input

#### Scenario: GradFn holds references to input tensors
- **WHEN** a `GradFn` is constructed with two input tensors `a`, `b`
- **THEN** `grad_fn.inputs` contains both `a` and `b`

### Requirement: `Tensor.backward()` accumulates gradients into leaf tensors

`tensor.backward()` SHALL assert the tensor is a scalar (shape `()`). It SHALL
seed the backward pass with `grad = np.ones(())`, then perform a reverse
topological sort of all `GradFn` nodes reachable from `tensor.grad_fn`. For each
node, it SHALL call `node.backward(accumulated_grad)` and distribute the resulting
gradient contributions. For leaf tensors, contributions SHALL be accumulated into
`.grad` (initialized to zeros on first touch). For non-leaf tensors, contributions
SHALL be held in a temporary buffer and passed to their `grad_fn` when that node
is processed.

#### Scenario: simple scalar backward populates leaf grad
- **GIVEN** leaf tensor `x` with `requires_grad=True` and value `3.0`
- **WHEN** `y = x * 2` (using a GradFn) and `y.backward()` is called
- **THEN** `x.grad` is a `Tensor` with `.item() == 2.0`

#### Scenario: diamond graph accumulates gradients correctly
- **GIVEN** leaf `x` with `requires_grad=True`, `a = f(x)`, `b = g(x)`, `loss = a + b`
- **WHEN** `loss.backward()` is called
- **THEN** `x.grad` equals the sum of `df/dx` and `dg/dx` (not just one of them)

#### Scenario: backward on non-scalar raises
- **WHEN** `tensor([1.0, 2.0]).backward()` is called (non-scalar)
- **THEN** an error is raised

### Requirement: `no_grad()` suppresses graph construction

`no_grad()` SHALL be a context manager. While active, all tensor operations SHALL
produce tensors with `grad_fn=None` regardless of whether inputs have
`requires_grad=True`. `requires_grad` on the output SHALL be `False`.

#### Scenario: operations inside no_grad produce no grad_fn
- **GIVEN** a tensor `x` with `requires_grad=True`
- **WHEN** `y = x * 2` is computed inside `with no_grad():`
- **THEN** `y.grad_fn is None` and `y.requires_grad is False`

#### Scenario: no_grad restores state on exit
- **GIVEN** code that exits a `no_grad()` context
- **WHEN** `y = x * 2` is computed after the `with` block
- **THEN** `y.grad_fn` is set normally (if `x.requires_grad`)

### Requirement: `Tensor.detach()` returns a tensor outside the graph

`tensor.detach()` SHALL return a new `Tensor` sharing the same underlying numpy
array but with `requires_grad=False` and `grad_fn=None`. Mutations to the
detached tensor's data SHALL reflect in the original (they share memory).

#### Scenario: detached tensor has no grad_fn
- **GIVEN** a non-leaf tensor `y` with `grad_fn` set
- **WHEN** `z = y.detach()`
- **THEN** `z.grad_fn is None` and `z.requires_grad is False`

#### Scenario: detached tensor shares data
- **GIVEN** `y` and `z = y.detach()`
- **THEN** `z.data` is the same numpy array as `y.data` (same identity)
