## Purpose

Specification for random Tensor creation functions (`rand`, `randint`) in Trainadillo, including generator seeding, value distribution, and reproducibility guarantees.

## Requirements

### Requirement: rand produces uniform float32 Tensors
`rand(*shape, generator=None)` SHALL return a float32 Tensor of the given shape with values drawn uniformly from [0, 1).

#### Scenario: Basic shape and dtype
- **WHEN** `rand(3, 4)` is called with a seeded generator
- **THEN** the result has shape `(3, 4)` and dtype `float32`

#### Scenario: Values in range
- **WHEN** `rand(1000)` is called
- **THEN** all values are in [0.0, 1.0)

#### Scenario: Reproducibility with same seed
- **WHEN** two generators are seeded with the same value and `rand` is called on each
- **THEN** both results are identical

#### Scenario: Different seeds diverge
- **WHEN** two generators are seeded with different values and `rand` is called on each
- **THEN** the results differ

### Requirement: randint produces int64 Tensors
`randint(low, high, size, *, generator=None)` SHALL return an int64 Tensor of shape `size` with values drawn uniformly from `[low, high)`. `size` SHALL be a tuple.

#### Scenario: Shape and dtype
- **WHEN** `randint(0, 10, (5,))` is called with a seeded generator
- **THEN** the result has shape `(5,)` and dtype `int64`

#### Scenario: Values in range
- **WHEN** `randint(0, 10, (1000,))` is called
- **THEN** all values satisfy `0 <= value < 10`

#### Scenario: Reproducibility with same seed
- **WHEN** two generators are seeded identically and `randint` is called on each
- **THEN** both results are identical

### Requirement: Unseeded default generator raises
When `generator=None` and `trainadillo.manual_seed()` has not been called, `rand` and `randint` SHALL raise `RuntimeError` with a message directing the user to call `trainadillo.manual_seed(seed)`.

#### Scenario: Raise before manual_seed
- **WHEN** `rand(3)` is called with no explicit generator and `manual_seed` has not been called
- **THEN** a `RuntimeError` is raised

#### Scenario: Succeeds after manual_seed
- **WHEN** `trainadillo.manual_seed(42)` is called and then `rand(3)` is called with no explicit generator
- **THEN** a valid Tensor is returned without error

### Requirement: Explicit generator bypasses default
When a `Generator` instance is passed explicitly, `rand` and `randint` SHALL use it regardless of whether the default generator has been seeded.

#### Scenario: Explicit generator with unseeded default
- **WHEN** `rand(3, generator=g)` is called with a seeded `g` and no `manual_seed` has been called
- **THEN** a valid Tensor is returned without error
