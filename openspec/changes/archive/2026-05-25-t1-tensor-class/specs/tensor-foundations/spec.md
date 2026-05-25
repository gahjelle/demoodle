# Spec: Tensor Foundations

## Purpose

Defines the `Tensor` type and `Size` type that form the foundation of trainadillo.
`Tensor` is a `numpy.ndarray` wrapper exposing a PyTorch-compatible public API.
In T1 it carries only data; T8 adds autograd fields (`grad`, `requires_grad`, `_grad_fn`, `_is_leaf`).

## Requirements

### Requirement: `Tensor` wraps a numpy array and exposes shape metadata

`Tensor` SHALL hold a `numpy.ndarray` internally and expose `.shape -> Size`, `.ndim -> int`, `.dtype -> np.dtype`, and `.data -> np.ndarray`.

#### Scenario: shape and ndim match the underlying array
- **GIVEN** a `Tensor` wrapping a `(3, 4)` numpy array
- **THEN** `.shape == Size([3, 4])` and `.ndim == 2`

### Requirement: `Size` is an immutable int-tuple with `numel()` and PyTorch repr

`Size` SHALL be a `tuple` subclass. `repr(Size([3, 4]))` SHALL equal `"torch.Size([3, 4])"` exactly. `Size([3, 4]).numel()` SHALL return `12`.

#### Scenario: Size repr matches torch.Size
- **WHEN** `repr(Size([3, 4]))` is evaluated
- **THEN** the result is `"torch.Size([3, 4])"`

#### Scenario: numel returns the product of dims
- **WHEN** `Size([2, 3, 4]).numel()` is called
- **THEN** the result is `24`

### Requirement: `item()` and `tolist()` extract Python values

`item()` SHALL return a Python scalar (`float`, `int`, or `bool`). `tolist()` SHALL return a Python list (or scalar for 0-D). Both delegate to numpy.

### Requirement: `size(dim=None)` returns `Size` or `int`

`size()` with no argument returns `.shape`. `size(dim)` returns `shape[dim]` as a Python `int`.

### Requirement: `view()` reshapes or reinterprets based on argument type

`tensor.view(*ints)` SHALL reshape (equivalent to `numpy.reshape`). `tensor.view(dtype)` where `dtype` is a numpy dtype SHALL reinterpret raw bytes (equivalent to `numpy.ndarray.view(dtype)`).

#### Scenario: view with int args reshapes
- **GIVEN** a 1-D Tensor of 12 float32 elements
- **WHEN** `.view(3, 4)` is called
- **THEN** the result has shape `(3, 4)`

#### Scenario: view with dtype reinterprets bytes
- **GIVEN** a 1-D Tensor of 3 float32 elements (12 bytes)
- **WHEN** `.view(np.uint8)` is called
- **THEN** the result has shape `(12,)` and dtype `uint8`

### Requirement: integer indexing returns a 0-D Tensor

`tensor[i]` where `i` is an integer SHALL return a `Tensor` with shape `()`, not a Python scalar. Slice indexing SHALL return a Tensor with the corresponding shape.

#### Scenario: integer index on 1-D tensor produces 0-D Tensor
- **GIVEN** a 1-D Tensor `t` of length 5
- **WHEN** `t[2]` is evaluated
- **THEN** the result is a `Tensor` with `.shape == Size([])` (0-D)

### Requirement: arithmetic operators return Tensors

`__add__`, `__sub__`, `__mul__`, `__truediv__`, `__neg__`, `__matmul__`, and their reflected variants (`__radd__`, etc.) SHALL all return `Tensor`. Mixed Tensor–scalar operations (e.g. `tensor + int`) SHALL work in both orders.

#### Scenario: scalar on left works via reflected operator
- **WHEN** `3 * tensor` is evaluated (int on left)
- **THEN** the result is a `Tensor` with values `3 × original`

### Requirement: comparison operators return boolean Tensors

`__gt__`, `__ge__`, `__lt__`, `__le__`, `__eq__` SHALL return a `Tensor` with boolean dtype. The result SHALL be usable as a mask (passed to indexing or `masked_fill`).

#### Scenario: comparison produces boolean Tensor
- **WHEN** `tensor > 0` is evaluated on a float Tensor
- **THEN** the result is a `Tensor` with `dtype == np.bool_`

### Requirement: `__bool__` is not implemented

Attempting `bool(tensor)` on a multi-element `Tensor` SHALL raise `TypeError`.

#### Scenario: bool conversion raises TypeError
- **WHEN** `bool(tensor)` is called on a Tensor with more than one element
- **THEN** `TypeError` is raised

### Requirement: dtype constants are module-level numpy dtype objects

`trainadillo.long`, `trainadillo.uint8`, `trainadillo.float32` SHALL be `np.int64`, `np.uint8`, `np.float32` respectively. Passing them as the `dtype` argument to `Tensor` construction SHALL produce a Tensor with the corresponding numpy dtype.

### Requirement: `__repr__` produces `tensor(...)` format

`repr(tensor)` SHALL start with `"tensor("` and end with `")"`. Float tensors SHALL display approximately 4 decimal places.
