## ADDED Requirements

### Requirement: `Tensor` carries autograd fields

`Tensor` SHALL have three instance fields added in T8:
- `grad: Tensor | None` — accumulated gradient; `None` until `backward()` populates it
- `requires_grad: bool` — whether this tensor participates in the computation graph
- `grad_fn: GradFn | None` — the function that created this tensor; `None` for leaves

These SHALL be public attributes (no underscore prefix), matching PyTorch's API.

#### Scenario: newly created Tensor has default autograd state
- **WHEN** `Tensor(np.array([1.0, 2.0]))` is constructed
- **THEN** `tensor.grad is None`, `tensor.requires_grad is False`, `tensor.grad_fn is None`

#### Scenario: requires_grad can be set at construction
- **WHEN** a Tensor is constructed with `requires_grad=True`
- **THEN** `tensor.requires_grad is True` and `tensor.grad_fn is None`

### Requirement: `is_leaf` is a derived property

`Tensor.is_leaf` SHALL be a `@property` returning `self.grad_fn is None`. A tensor
is a leaf if and only if it was created directly (not by an operation). There SHALL
be no separate stored `_is_leaf` field.

#### Scenario: user-created tensor is a leaf
- **GIVEN** `x = Tensor(np.array(1.0))` with or without `requires_grad=True`
- **THEN** `x.is_leaf is True`

#### Scenario: operation output is not a leaf
- **GIVEN** a tensor `y` with `grad_fn` pointing to a `GradFn` instance
- **THEN** `y.is_leaf is False`

## MODIFIED Requirements

### Requirement: `view()` reshapes or reinterprets based on argument type

`tensor.view(*ints)` SHALL reshape (equivalent to `numpy.reshape`). `tensor.view(dtype)`
where `dtype` is a numpy dtype SHALL reinterpret raw bytes (equivalent to
`numpy.ndarray.view(dtype)`). The return type SHALL always be `Tensor`, not
`type(self)` — a view of a `Parameter` is a plain `Tensor`.

#### Scenario: view with int args reshapes
- **GIVEN** a 1-D Tensor of 12 float32 elements
- **WHEN** `.view(3, 4)` is called
- **THEN** the result has shape `(3, 4)` and is a `Tensor` instance

#### Scenario: view with dtype reinterprets bytes
- **GIVEN** a 1-D Tensor of 3 float32 elements (12 bytes)
- **WHEN** `.view(np.uint8)` is called
- **THEN** the result has shape `(12,)` and dtype `uint8`

#### Scenario: view on a subclass returns Tensor, not the subclass
- **GIVEN** `p` is an instance of a `Tensor` subclass
- **WHEN** `p.view(1, -1)` is called
- **THEN** the result is a `Tensor`, not a subclass instance

### Requirement: `squeeze()` and `flatten()` return `Tensor`

`squeeze()` and `flatten()` SHALL always return a plain `Tensor`, not `type(self)`.
This ensures subclasses (e.g. `Parameter`) do not leak through operations that
produce derived tensors. `cpu()` and `contiguous()` are no-ops that return `self`
unchanged — returning `self` is correct for these methods and does not produce
a new derived tensor.

#### Scenario: squeeze on subclass returns Tensor
- **GIVEN** `p` is a `Tensor` subclass instance with a size-1 dimension
- **WHEN** `p.squeeze()` is called
- **THEN** the result is a `Tensor`, not a subclass instance

#### Scenario: flatten on subclass returns Tensor
- **GIVEN** `p` is a `Tensor` subclass instance with shape `(2, 3)`
- **WHEN** `p.flatten()` is called
- **THEN** the result is a `Tensor`, not a subclass instance
