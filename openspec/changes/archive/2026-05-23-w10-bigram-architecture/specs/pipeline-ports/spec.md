## MODIFIED Requirements

### Requirement: InspectableProtocol defines the inference and inspection interface
The system SHALL provide an `InspectableProtocol` where `call(seq: Seq, policy: Policy, rng: RNG, temperature: float, top_k: int | None = None, top_p: float | None = None) -> Output` is required and `explain(seq: Seq, policy: Policy) -> dict[str, Any]` is optional with a default body returning `{}`. Implementations that do not override `explain` inherit the default.

Both `call` and `explain` receive `policy` explicitly. Architectures are stateless config/logic; all model state lives in `Policy`. This mirrors `ArchitectureProtocol.forward(policy, tokens)` — no architecture holds a `Policy` reference internally.

`call` returns an `Output` containing both `logits` (the full distribution over the vocabulary) and `sampled_ids` (the single drawn token id), enabling front ends to visualise the probability distribution without a second forward pass.

`top_k` and `top_p` are optional sampling controls. Implementations MAY ignore them when not applicable (e.g. at small vocabulary sizes where they have no meaningful effect), but SHALL accept them without error.

#### Scenario: Dummy class with only `call` satisfies InspectableProtocol
- **WHEN** a class implements only `call(seq, policy, rng, temperature, top_k=None, top_p=None)`
- **THEN** it type-checks as a valid `InspectableProtocol` and `explain(seq, policy)` returns `{}`

#### Scenario: Overriding `explain` returns custom data
- **WHEN** an implementation overrides `explain(seq, policy)` to return attention weights
- **THEN** calling `explain(seq, policy)` returns that implementation's data, not `{}`

#### Scenario: call receives policy explicitly
- **WHEN** a conforming implementation's `call` is invoked
- **THEN** it receives the model state via `policy`, not from any internal field on the architecture
