# Core Foundation: Types, RNG, and Protocols (W2–W4)

This file captures design decisions and gotchas from the first three substantive work items. Read it before implementing anything that touches `core/` or `ports/`.

---

## Artifacts vs. RNG vs. Protocols — three distinct roles

```
core/types.py      Artifact union — values that flow between stages and get cached
core/rng.py        RNG — threading infrastructure passed into stages, never cached
ports/protocols.py Protocols + Stage — behavioral contracts and the stage definition
```

Do not conflate them. `RNG` is not an `Artifact`. Protocols are not artifacts. The `Artifact` type union is the complete list of what stages produce and consume.

---

## Naming: artifact vs. protocol

Every protocol has a `Protocol` suffix to avoid shadowing the artifact type of the same name:

| Artifact (`core.types`) | Protocol (`ports.protocols`) |
|-------------------------|------------------------------|
| `Tokenizer`             | `TokenizerProtocol`          |
| *(no artifact)*         | `ArchitectureProtocol`       |
| *(no artifact)*         | `InspectableProtocol`        |

Import site is unambiguous:
- `from demoodle.core.types import Tokenizer` → the data artifact (just `vocab_size`)
- `from demoodle.ports.protocols import TokenizerProtocol` → the behavioral contract

---

## `InspectableProtocol.explain` — default body requires inheritance

`explain()` has a default body returning `{}` in the Protocol class. This default is **only inherited if a class explicitly subclasses `InspectableProtocol`**. Structural implementors (classes that implement the right methods without inheriting) do not get the default:

```python
# Gets the default explain():
class MyArch(InspectableProtocol):
    def call(self, seq, temperature): ...

# Does NOT get explain() — AttributeError at runtime:
class MyArch:
    def call(self, seq, temperature): ...
```

Architectures that don't implement `explain` (bigram, MLP) should subclass `InspectableProtocol` to inherit the no-op default. Architectures that do implement it (transformer, W15) can either subclass or define it independently.

---

## RNG threading — how it works at stage boundaries

The runner (W6) splits the RNG before every stage call:

```python
stage_rng, rng = rng.split()
outputs = stage.run(artifact_dict, stage_rng)
```

Stages that don't need randomness ignore `_rng`. Stages that do use `rng.generator()` to get a seeded `torch.Generator`. Never use `torch.manual_seed()` or any global seed — pass the generator explicitly to PyTorch ops.

Three-way splits chain two calls: `a, tmp = rng.split(); b, c = tmp.split()`.

---

## `Dataset.tokens` is a flat sequence — windowing is the architecture's job

`build_dataset` (W9) produces a flat 1D token tensor. It cannot produce batches because it has no access to the architecture config (context window size). Each architecture's training loop slices the flat tensor into `[N, context_len]` windows. This keeps `build_dataset` architecture-agnostic.

---

## `Policy` holds a live `nn.Module` — write-once by convention

`Policy` is a frozen dataclass (field reassignment raises). But the embedded `nn.Module` is mutable — PyTorch requires it for gradient computation. The invariant is semantic: training always produces a **new** `Policy` rather than mutating an existing one. No stage should call `.train()`, `.zero_grad()`, or mutate weights on a `Policy` it did not just create.

---

## `Tokenizer` artifact will be extended by W8

The `Tokenizer` artifact in `core/types.py` currently holds only `vocab_size: int`. `CharTokenizer` (W8) will subclass it. When that happens, the `Artifact` union (`core/types.py`, bottom of file) is the single place to update if the union member changes. The `TokenizerProtocol` in `ports/protocols.py` is separate and not affected.

---

## `Stage` fields use `list[str]`, not `tuple`

`needs` and `produces` are `list[str]` per the project convention: `list` for homogeneous sequences, `tuple` for heterogeneous positional records. Treat them as read-only; do not mutate after construction.
