# Core Foundation: Types, RNG, Protocols, and Runner (W2–W6)

This file captures design decisions and gotchas from the first three substantive work items. Read it before implementing anything that touches `core/` or `ports/`.

---

## No `from __future__ import annotations`

This project targets Python 3.14, which evaluates annotations lazily by default (PEP 649). Do **not** add `from __future__ import annotations` to any file — it is unnecessary and being phased out.

For `TYPE_CHECKING`-only imports, keep the `if TYPE_CHECKING:` block and use the names unquoted in annotations; the runtime never evaluates them.

```python
# correct
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from demoodle.core.rng import RNG

def foo(rng: RNG) -> None: ...

# wrong — do not add this
from __future__ import annotations
```

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

| Artifact (`core.types`)           | Protocol (`ports.protocols`) |
|-----------------------------------|------------------------------|
| `CharTokenizer`, `BpeTokenizer`   | `TokenizerProtocol`          |
| *(no artifact)*                   | `ArchitectureProtocol`       |
| *(no artifact)*                   | `InspectableProtocol`        |

Import site is unambiguous:
- `from demoodle.tokenizers.char import CharTokenizer` → the concrete tokenizer artifact
- `from demoodle.ports.protocols import TokenizerProtocol` → the behavioral contract

---

## `InspectableProtocol.explain` — stateless, default body requires inheritance

`explain(seq: Seq) -> dict[str, Any]` is a pure function: it takes the same context sequence as `call` and re-runs inference internally to produce arch-specific interpretability data (e.g. attention weights). No mutable internal buffers; no implicit ordering requirement relative to `call`.

The default body returns `{}` and is **only inherited if a class explicitly subclasses `InspectableProtocol`**. Structural implementors (classes that implement the right methods without inheriting) do not get the default:

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

## Stages do not validate their inputs at runtime

Stage `run` functions trust the artifact dict they receive. They do not `isinstance`-check their inputs. The runner guarantees that a stage only executes when all its `needs` are satisfied, and the type system enforces correctness statically. Adding runtime isinstance checks would couple `tokenizers/` back to `core/types.py`, creating a circular import.

Use a narrowing annotation instead of a check:

```python
def run(artifacts: dict[str, Artifact], _rng: RNG) -> dict[str, Artifact]:
    corpus: Corpus = artifacts["corpus"]  # ty: ignore[invalid-assignment]
    ...
```

---

## Tokenizer artifacts — concrete types go in the `Artifact` union (W8, W17)

There is no shared `Tokenizer` base artifact. Instead, concrete tokenizer types are added to the `Artifact` union directly:

- **W8**: `CharTokenizer` frozen dataclass added; the `Tokenizer` placeholder removed.
- **W17**: `BpeTokenizer` frozen dataclass added.

Both satisfy `TokenizerProtocol` structurally — no inheritance. Stages receive whichever concrete type is in the artifact dict and call `encode`/`decode`/`vocab_size` through the protocol. `_hash_artifact` in `shell/persistence.py` gets one branch per concrete type.

The `Tokenizer` placeholder was removed in W8 when `CharTokenizer` was added.

---

## `Stage.config_hash` — required, no default (added W5)

`Stage` has a required `config_hash: str` field with no default. Every stage construction must supply it explicitly. The value should be a hash of all pydantic sub-configs that affect the stage's output:

```python
import hashlib
config_hash = hashlib.sha256(training_cfg.model_dump_json().encode()).hexdigest()
Stage(name="pretrain", ..., config_hash=config_hash, run=run)
```

Stages whose output is independent of all config pass `config_hash=""` explicitly — this is intentional documentation, not an omission. The cache key (`shell/persistence.py`) combines this hash with the stage name, code version, input artifact hashes, and RNG seed.

---

## `ArchitectureProtocol.init_state` — config bound at construction (W10)

`init_state(self, rng: RNG) -> Policy`. The architecture object holds its own hyperparameters (vocab size, hidden size, etc.) — it is the configured instance. Config gets bound when the architecture is constructed, not when `init_state` is called:

```python
bigram = BigramArchitecture(vocab_size=27)
policy = bigram.init_state(rng)   # pure function of rng only
```

This keeps `init_state` a pure function of `rng` and avoids threading config through the stage's artifact dict.

---

## Runner public API — `shell/runner.py`

`run(stages, initial_artifacts, rng, cache_dir) → dict[str, Artifact]`

- `stages`: list of `Stage`s in any order — the runner topo-sorts them.
- `initial_artifacts`: dict of seed artifacts (e.g. a loaded `Corpus`) that no stage produces.
- `rng`: root `RNG`; the runner splits it once per stage in execution order, unconditionally.
- `cache_dir`: `Path` to the content-addressed cache directory.

Returns all artifacts: both the initial dict and every stage output merged together.

Raises `ValueError` before any execution if:
- Two stages declare the same artifact name in `produces`.
- The graph is unsatisfiable (cycle or genuinely missing artifact) — error message includes each stuck stage name and the artifact names it's waiting for.

---

## `Stage` fields use `list[str]`, not `tuple`

`needs` and `produces` are `list[str]` per the project convention: `list` for homogeneous sequences, `tuple` for heterogeneous positional records. Treat them as read-only; do not mutate after construction.
