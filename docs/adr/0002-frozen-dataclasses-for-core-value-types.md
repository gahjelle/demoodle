# ADR-0002: Frozen dataclasses for core value types, not Pydantic

**Status:** Accepted

## Context

Pydantic is already a project dependency, used for configuration (`DemoodleConfig`
and section models in `config/schemas.py`). When defining the pipeline's value types
(`Corpus`, `Dataset`, `Policy`, `Metrics`, …) the question arose whether to use the
same library or a different mechanism.

## Decision

Use `@dataclass(frozen=True)` from the standard library for all types in `core/types.py`.

## Reasoning

Pydantic's value proposition is validation and coercion at system boundaries — parsing
env vars, config files, and user input into well-typed Python objects. The core value
types have none of those concerns: they are internal pipeline values produced by one
stage and consumed by the next. No validation rules, no field coercion, no JSON
parsing.

`@dataclass(frozen=True)` is lighter (no extra dependency beyond what's in the
stdlib), faster to construct, and signals "this is pure data" to a reader. Pydantic
models signal "this is a validated boundary type."

The rule of thumb this establishes: **Pydantic is for config; frozen dataclasses are
for core.**

## Consequences

- All artifact types in `core/types.py` are `@dataclass(frozen=True)`
- Field reassignment raises `FrozenInstanceError` at runtime
- The write-once semantic for `Policy` (training produces a new `Policy`, never
  mutates an existing one) is enforced by the frozen dataclass, not by convention alone
- `nn.Module` inside `Policy` is itself mutable — `frozen=True` prevents replacing
  the field, not mutating the module's weights. This is intentional: the frozen
  contract is about the artifact reference, not deep immutability of PyTorch internals
