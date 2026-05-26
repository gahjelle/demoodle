## ADDED Requirements

### Requirement: `Tensor.argmax(dim=None)` returns the index of the maximum

`Tensor.argmax(dim=None)` SHALL return a `Tensor` containing the index of the
maximum element. When `dim` is `None`, the operation is over the flattened
tensor and the result is a 0-D Tensor. When `dim` is given, the result has
that dimension removed (one index per slice along `dim`). Matches
`torch.Tensor.argmax` behaviour.

#### Scenario: argmax over flat tensor returns 0-D Tensor
- **WHEN** `tensor([3.0, 1.0, 5.0, 2.0]).argmax()` is called
- **THEN** the result is a `Tensor` with shape `()` and `.item() == 2`

#### Scenario: argmax along dim=0 on a 2-D tensor
- **GIVEN** `t = tensor([[1.0, 5.0], [3.0, 2.0]])` (shape `(2, 2)`)
- **WHEN** `t.argmax(dim=0)` is called
- **THEN** the result has shape `(2,)` and values `[1, 0]`
  (column 0 max is at row 1; column 1 max is at row 0)

#### Scenario: argmax result is a Tensor, not a Python int
- **WHEN** `tensor([1.0, 2.0, 3.0]).argmax()` is called
- **THEN** the result is a `Tensor` instance, not a plain Python `int`
