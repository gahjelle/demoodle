## ADDED Requirements

### Requirement: tensor() constructs a Tensor from Python data
The `tensor(data, *, dtype=None)` function SHALL construct a new `Tensor` by copying
the input data. When `dtype` is `None` and the inferred NumPy dtype is `float64`,
the result SHALL be cast to `float32`. When `dtype` is given explicitly, it overrides
inference. The function SHALL accept Python lists, scalars, NumPy arrays, and existing
`Tensor` objects as input, and SHALL always copy the underlying data.

#### Scenario: List of ints produces int64 Tensor
- **WHEN** `tensor([1, 2, 3])` is called with no dtype
- **THEN** the result is a `Tensor` with dtype `int64` and values `[1, 2, 3]`

#### Scenario: List of floats produces float32 Tensor
- **WHEN** `tensor([1.0, 2.0, 3.0])` is called with no dtype
- **THEN** the result is a `Tensor` with dtype `float32` (not `float64`)

#### Scenario: Explicit dtype overrides inference
- **WHEN** `tensor([1, 2, 3], dtype=float32)` is called
- **THEN** the result is a `Tensor` with dtype `float32`

#### Scenario: Existing Tensor input is copied
- **WHEN** `tensor(existing_tensor)` is called
- **THEN** the result is a new `Tensor` with the same values
- **AND** mutating the original does not affect the copy

### Requirement: zeros() and ones() create constant Tensors
`zeros(*shape)` SHALL return a `Tensor` of the given shape filled with `0.0` and
dtype `float32`. `ones(*shape)` SHALL return a `Tensor` of the given shape filled
with `1.0` and dtype `float32`. Shape SHALL be passed as variadic arguments
(e.g. `zeros(3, 4)`, not `zeros((3, 4))`).

#### Scenario: zeros produces float32 zeros
- **WHEN** `zeros(3, 4)` is called
- **THEN** the result has shape `(3, 4)`, dtype `float32`, and all values are `0.0`

#### Scenario: ones produces float32 ones
- **WHEN** `ones(2, 3)` is called
- **THEN** the result has shape `(2, 3)`, dtype `float32`, and all values are `1.0`

### Requirement: zeros_like() and full_like() mirror an existing Tensor
`zeros_like(t)` SHALL return a `Tensor` with the same shape and dtype as `t`, filled
with zeros. `full_like(t, value)` SHALL return a `Tensor` with the same shape and
dtype as `t`, filled with `value`.

#### Scenario: zeros_like preserves dtype
- **WHEN** `zeros_like(t)` is called on an `int64` Tensor
- **THEN** the result has dtype `int64` and all values are `0`

#### Scenario: full_like preserves shape and dtype
- **WHEN** `full_like(t, 7)` is called on a `float32` Tensor of shape `(2, 3)`
- **THEN** the result has shape `(2, 3)`, dtype `float32`, and all values are `7.0`

### Requirement: arange() produces an integer range Tensor
`arange(n)` SHALL return a 1-D `Tensor` containing integers `[0, 1, ..., n-1]` with
dtype `int64` regardless of the platform's native integer width.

#### Scenario: arange returns int64
- **WHEN** `arange(5)` is called
- **THEN** the result has shape `(5,)`, dtype `int64`, and values `[0, 1, 2, 3, 4]`

### Requirement: stack() combines Tensors along a new axis
`stack(tensors, dim=0)` SHALL combine a sequence of Tensors of equal shape into a
new Tensor with one additional dimension inserted at `dim`. This is equivalent to
`np.stack`.

#### Scenario: stack of 1-D Tensors produces 2-D Tensor
- **WHEN** `stack([tensor([1, 2]), tensor([3, 4]), tensor([5, 6])])` is called
- **THEN** the result has shape `(3, 2)` and values `[[1,2],[3,4],[5,6]]`

#### Scenario: stack with dim=1 inserts axis at position 1
- **WHEN** `stack([tensor([1, 2]), tensor([3, 4])], dim=1)` is called
- **THEN** the result has shape `(2, 2)` and values `[[1,3],[2,4]]`

### Requirement: equal() reduces two Tensors to a Python bool
`equal(a, b)` SHALL return a Python `bool` that is `True` if and only if `a` and `b`
have the same shape and all corresponding elements are equal. This is a full reduction,
not an element-wise operation.

#### Scenario: equal Tensors return True
- **WHEN** `equal(tensor([1, 2, 3]), tensor([1, 2, 3]))` is called
- **THEN** the result is the Python value `True`

#### Scenario: unequal Tensors return False
- **WHEN** `equal(tensor([1, 2, 3]), tensor([1, 2, 4]))` is called
- **THEN** the result is the Python value `False`

#### Scenario: different shapes return False
- **WHEN** `equal(tensor([1, 2]), tensor([1, 2, 3]))` is called
- **THEN** the result is the Python value `False`

### Requirement: Tensor arithmetic with Python int is valid for indexing
A `Tensor` plus a Python `int` SHALL produce a `Tensor` (via `Tensor.__add__`), and
that result SHALL be usable as an index into another `Tensor`. This supports the
`tokens[offsets + context_len]` pattern in the model's data pipeline.

#### Scenario: Tensor index offset by Python int
- **WHEN** `tokens[offsets + context_len]` is evaluated where `offsets` is an `int64`
  Tensor and `context_len` is a Python `int`
- **THEN** the result is a `Tensor` containing the elements of `tokens` at the
  offset positions
