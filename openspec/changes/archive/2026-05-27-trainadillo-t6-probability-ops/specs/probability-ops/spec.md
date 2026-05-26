## ADDED Requirements

### Requirement: `softmax` returns a numerically stable probability distribution
`softmax(tensor, dim)` SHALL compute `exp(x - max(x)) / sum(exp(x - max(x)))` along
`dim` and return a `Tensor`. The output SHALL sum to 1.0 along `dim`. The function
SHALL accept any valid `dim` value, including negative indices.

#### Scenario: Output sums to 1
- **WHEN** `softmax(tensor, dim=-1)` is called on a 1-D float Tensor
- **THEN** the output values sum to 1.0 (within float32 tolerance)

#### Scenario: Large logits do not produce NaN or Inf
- **WHEN** `softmax` is called on a Tensor containing values like `1000.0` or `-1000.0`
- **THEN** the output contains no NaN or Inf values

#### Scenario: dim parameter selects the reduction axis
- **WHEN** `softmax(tensor, dim=0)` is called on a 2-D Tensor of shape `(R, C)`
- **THEN** each column of the output sums to 1.0

### Requirement: `cumsum` computes cumulative sum along a dimension
`cumsum(tensor, dim)` SHALL return a `Tensor` of the same shape as the input where
each element is the sum of all preceding elements (inclusive) along `dim`.

#### Scenario: cumsum matches numpy
- **WHEN** `cumsum(tensor, dim=-1)` is called on a 1-D Tensor
- **THEN** the result equals `np.cumsum(tensor._data, axis=-1)` element-wise

#### Scenario: Output shape equals input shape
- **WHEN** `cumsum(tensor, dim=0)` is called on a 2-D Tensor
- **THEN** the output shape equals the input shape

### Requirement: `multinomial` samples indices from a categorical distribution
`multinomial(probs, num_samples, *, generator=None)` SHALL return a 1-D `Tensor`
of shape `(num_samples,)` containing integer indices sampled according to the
probability weights in `probs`. When `generator` is `None`, the module-level
default generator SHALL be used.

#### Scenario: Output shape is (num_samples,)
- **WHEN** `multinomial(probs, 3)` is called on a 1-D Tensor of length V
- **THEN** the result has shape `(3,)` and dtype int64

#### Scenario: Same generator seed produces the same samples
- **GIVEN** two `Generator` instances both seeded with the same integer
- **WHEN** `multinomial(probs, num_samples=1, generator=g)` is called on each
- **THEN** both calls return the same index

#### Scenario: Different seeds produce different samples
- **GIVEN** two `Generator` instances seeded with different integers
- **WHEN** `multinomial(probs, num_samples=1, generator=g)` is called on each
- **THEN** the results differ (with overwhelming probability for well-spread probs)

#### Scenario: Sampling frequencies approximate input probabilities
- **WHEN** `multinomial` is called 10 000 times with the same `probs` and independent seeds
- **THEN** the frequency of each index is within a reasonable tolerance of its probability weight
