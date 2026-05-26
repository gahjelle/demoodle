# Spec: tensor-masking-ops

## Purpose

Masking and scatter operations on Tensor that support attention-mask construction, token filtering (top-k/top-p), and probability pipeline validation.

## Requirements

### Requirement: masked_fill replaces masked positions with a fill value
`Tensor.masked_fill(mask, value)` SHALL return a new Tensor of the same shape and dtype as `self`, where every position where `mask` is True is replaced with `value`, and all other positions retain their original values. `self` SHALL NOT be modified.

#### Scenario: True positions are replaced
- **WHEN** `masked_fill` is called with a boolean mask and a fill value
- **THEN** every position where the mask is True contains the fill value in the result

#### Scenario: False positions are unchanged
- **WHEN** `masked_fill` is called with a boolean mask and a fill value
- **THEN** every position where the mask is False retains its original value from `self`

#### Scenario: self is not mutated
- **WHEN** `masked_fill` is called
- **THEN** `self._data` is unchanged after the call

#### Scenario: fill value is float negative infinity
- **WHEN** `masked_fill` is called with `value=float("-inf")`
- **THEN** the result at masked positions contains negative infinity (used to eliminate tokens before softmax)

### Requirement: scatter writes src values at index positions
`Tensor.scatter(dim, index, src)` SHALL return a new Tensor equal to `self` with values from `src` written at positions given by `index` along `dim`. For `dim=0` on a 1-D tensor, position `index[i]` in the result is set to `src[i]`. `self` SHALL NOT be modified. `src` SHALL be a Tensor (not a scalar).

#### Scenario: values are placed at specified indices
- **WHEN** `scatter` is called with `dim=0`, a 1-D index tensor, and a 1-D src tensor
- **THEN** `result[index[i]] == src[i]` for all `i`

#### Scenario: unindexed positions retain original values
- **WHEN** `scatter` is called and some positions are not referenced by `index`
- **THEN** those positions in the result equal the corresponding positions in `self`

#### Scenario: self is not mutated
- **WHEN** `scatter` is called
- **THEN** `self._data` is unchanged after the call

#### Scenario: scatter reverses a sort permutation
- **WHEN** `scatter` is called with `zeros_like(x)` as self, `sorted_indices` as index, and sorted values as src
- **THEN** the result is the original unsorted tensor (scatter inverts the sort permutation)

### Requirement: _sample() filtering pipeline matches PyTorch
The `_sample()` function in `bigram.py` SHALL produce the same probability tensor (before the final `multinomial` draw) as an equivalent PyTorch implementation, to within float32 precision, for all combinations of `top_k` and `top_p` filtering.

#### Scenario: deterministic corner case top_k=1 agrees exactly
- **WHEN** `_sample()` is called with `top_k=1`
- **THEN** both trainadillo and PyTorch implementations return the argmax token (no randomness; exact agreement)

#### Scenario: deterministic corner case top_p=0.0 agrees exactly
- **WHEN** `_sample()` is called with `top_p=0.0`
- **THEN** both trainadillo and PyTorch implementations return the argmax token (no randomness; exact agreement)

#### Scenario: probs tensor matches PyTorch before multinomial
- **WHEN** the filtering pipeline up to (not including) `multinomial` is run with identical logits through both libraries
- **THEN** the resulting probability tensors are element-wise close (`np.allclose` with `rtol=1e-5`) for top_k, top_p, and combined filtering
