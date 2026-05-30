## MODIFIED Requirements

### Requirement: Sampling helper is available as a module-level pure function
The system SHALL provide `sample(logits, temperature, top_k, top_p, generator) -> torch.Tensor` in `demoodle.architectures.sampling` (not in `demoodle.architectures.bigram`). It SHALL be a pure function: given the same inputs and generator state, it returns the same token id. `demoodle.architectures.bigram` SHALL import and use `sample` from `demoodle.architectures.sampling`.

#### Scenario: sample with temperature=1 and no filters samples from full distribution
- **WHEN** `sample` is called with `top_k=None, top_p=None, temperature=1.0`
- **THEN** any token id in `[0, vocab_size)` can be returned (no hard exclusions)

#### Scenario: sample with top_k=1 always returns argmax
- **WHEN** `sample` is called with `top_k=1`
- **THEN** the returned tensor equals `logits.argmax()`
