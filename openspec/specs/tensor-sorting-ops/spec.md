# Spec: Tensor Sorting Operations

## Purpose

Defines sorting and top-k selection operations on `Tensor`. These operations
return named-tuple results carrying both `values` and `indices`, mirroring the
PyTorch `torch.topk` / `torch.sort` API.

## Requirements

### Requirement: `topk` returns the k largest values and their source indices

`topk(tensor, k, dim=-1)` SHALL return a `TopKResult` NamedTuple with fields
`values` and `indices`, both `Tensor`. `values` SHALL contain the k largest
elements along `dim`, in descending order (largest first). `indices` SHALL
contain the positions those elements occupied in the original tensor. The return
value SHALL support both positional destructuring (`values, indices = topk(...)`)
and named access (`.values`, `.indices`).

#### Scenario: top-1 of a 1-D tensor
- **WHEN** `topk(tensor([3.0, 1.0, 4.0, 1.0, 5.0]), k=1)` is called
- **THEN** `result.values.item() == 5.0` and `result.indices.item() == 4`

#### Scenario: top-3 returns descending order
- **WHEN** `topk(tensor([3.0, 1.0, 4.0, 1.0, 5.0]), k=3)` is called
- **THEN** `result.values.tolist() == [5.0, 4.0, 3.0]`
- **THEN** `result.indices.tolist() == [4, 2, 0]`

#### Scenario: k equals tensor length
- **WHEN** `topk(t, k=len(t))` is called
- **THEN** the result is a full descending sort of `t`

#### Scenario: named access works
- **WHEN** `result = topk(t, k=2)`
- **THEN** `result.values` and `result.indices` are both `Tensor`

### Requirement: `sort` returns fully sorted values and the permutation indices

`sort(tensor, dim=-1, *, descending=False)` SHALL return a `SortResult`
NamedTuple with fields `values` and `indices`. `values` SHALL be the elements of
`tensor` sorted along `dim`. `indices` SHALL be the permutation that maps sorted
positions back to original positions (i.e. `tensor[result.indices] == result.values`
for 1-D tensors). `descending` SHALL be keyword-only.

#### Scenario: ascending sort of 1-D tensor
- **WHEN** `sort(tensor([3.0, 1.0, 2.0]))` is called
- **THEN** `result.values.tolist() == [1.0, 2.0, 3.0]`
- **THEN** `result.indices.tolist() == [1, 2, 0]`

#### Scenario: descending sort
- **WHEN** `sort(tensor([3.0, 1.0, 2.0]), descending=True)` is called
- **THEN** `result.values.tolist() == [3.0, 2.0, 1.0]`
- **THEN** `result.indices.tolist() == [0, 2, 1]`

#### Scenario: `descending` is keyword-only
- **WHEN** `sort(t, -1, True)` is called (positional `descending`)
- **THEN** `TypeError` is raised

#### Scenario: named access works
- **WHEN** `result = sort(t)`
- **THEN** `result.values` and `result.indices` are both `Tensor`

### Requirement: indices from `sort` recover original values via indexing

For any 1-D Tensor `t`, after `values, indices = sort(t)`, indexing `t` with
the sort indices SHALL recover the sorted values.

#### Scenario: round-trip via indices
- **GIVEN** `t = tensor([3.0, 1.0, 4.0])` and `values, indices = sort(t)`
- **WHEN** `t[indices]` is evaluated
- **THEN** the result equals `values` element-wise
