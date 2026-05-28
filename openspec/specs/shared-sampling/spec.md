# Spec: Shared Sampling

## Purpose

TBD — documents the shared `sample` function in `demoodle.architectures.sampling` that is used by all architecture modules rather than each defining its own sampling logic.

## Requirements

### Requirement: Shared sampling function available in architectures.sampling
The system SHALL provide a public function `sample(logits, temperature, top_k, top_p, generator) -> torch.Tensor` in `demoodle.architectures.sampling`. It SHALL be a pure function: given the same inputs and generator state, it returns the same token id. Architecture modules SHALL import `sample` from this module rather than defining their own.

#### Scenario: sample with top_k=1 always returns argmax
- **WHEN** `sample` is called with `top_k=1` and any logits tensor
- **THEN** the returned tensor equals `logits.argmax()`

#### Scenario: sample with top_p=0.0 always returns argmax
- **WHEN** `sample` is called with `top_p=0.0` and any logits tensor
- **THEN** the returned tensor equals `logits.argmax()`

#### Scenario: sample with no filters samples from full distribution
- **WHEN** `sample` is called with `top_k=None, top_p=None, temperature=1.0` and uniform logits over many calls
- **THEN** multiple distinct token ids are returned (no hard exclusions)

#### Scenario: sample is importable from the architectures package
- **WHEN** `from demoodle.architectures.sampling import sample` is executed
- **THEN** the import succeeds and the function is callable
