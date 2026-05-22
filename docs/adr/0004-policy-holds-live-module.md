# ADR-0004: Policy holds a live nn.Module, not a state_dict

**Status:** Accepted

## Context

`Policy` is the artifact produced by training stages and consumed by generation
and inspection. The question was whether `Policy` should hold a live
`torch.nn.Module` (ready to call) or a `state_dict` dict (the functionally purer
form that carries only weights).

## Decision

`Policy.model` holds a live `torch.nn.Module`.

## Reasoning

A `state_dict`-only `Policy` would require reconstructing the module architecture
on every forward pass — callers would need to know the architecture class and
config to do so. This pushes architecture knowledge into every consumer of
`Policy` (the generation loop, `explain()`, the CLI, the TUI). It also makes
inference loops expensive.

The live module approach means callers can call `policy.model(tokens)` directly
without knowing anything about the architecture. This is the right abstraction for
the shell: the shell calls the model, it does not reconstruct it.

The `Policy` frozen dataclass prevents field reassignment (`policy.model = new_model`
raises `FrozenInstanceError`). The write-once semantic — training produces a new
`Policy` rather than mutating an existing one — is enforced at the reference level.
The `nn.Module`'s internal weights are still mutable (PyTorch's design), but no
stage in this pipeline mutates a `Policy` after creation.

The `value_head` field (`nn.Module | None`, default `None`) is reserved for PPO
(W27). It is defined on `Policy` now so the architecture never needs to be
retrofitted when the PPO stage is added.

## Consequences

- Forward passes and `explain()` calls are simple: `policy.model(tokens)`
- `torch.save` / `torch.load` handles `nn.Module` serialization in the artifact
  cache — no custom serialization needed
- Deep immutability of weights is not enforced; convention and code review are the
  safeguard against accidental mutation
- The architecture class is not stored in `Policy` — callers that need to
  reconstruct the model (e.g. to change device) must retain the architecture object
  separately
