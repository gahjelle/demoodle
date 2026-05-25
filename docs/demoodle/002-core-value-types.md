# W2 — Core Value Types

## Functional Core / Imperative Shell

Demoodle is organized around one architectural principle: the **functional core**
contains only pure functions and immutable values; the **imperative shell** handles
I/O, caching, and side effects.

The functional core includes `core/`, `ports/`, `architectures/`, `tokenizers/`,
`training/`, and `data/`. The imperative shell is `shell/` and `frontends/`.

The payoff is testability. A pure function from `Artifact` to `Artifact` requires
no filesystem, no RNG state, no global config to set up. The shell is thin enough
that its correctness is obvious by inspection.

Real ML systems that have rediscovered this split:
- **JAX**: all array operations are pure functions; state lives in parameter
  pytrees managed by Haiku or Flax
- **HuggingFace Datasets**: processing pipelines are chains of pure
  transformations applied lazily over Arrow tables
- **MLflow**: the tracking API records side effects (metrics, artifacts) through
  an explicit client; model logic stays pure

## Immutable values as pipeline data

Every type in `core/types.py` is a frozen dataclass. Frozen means `__setattr__`
raises after construction.

Immutability is what makes caching correct. If an artifact is immutable, its hash
is stable from the moment it's created. A mutable artifact could be modified in
place after one stage produces it and before the next stage consumes it —
silently corrupting the cache key of everything downstream.

Immutability also makes data flow explicit. Reading a stage's `needs` and
`produces` lists tells you exactly what it consumes and creates. Nothing is passed
by reference and mutated silently.

## The `Artifact` tagged union

```python
type Artifact = Corpus | CharTokenizer | Dataset | Policy | TrainingMetrics
```

Every value that flows between stages is one of these types — a **tagged union**
(also called a sum type or discriminated union). A value is exactly one variant,
and Python's `match` statement can dispatch on it exhaustively.

HuggingFace has a similar concept: `BaseModelOutput` and its subclasses form a
structured output hierarchy where each field is explicitly typed. MLflow's artifact
model defines typed "flavors" (sklearn, pytorch, python_function) with the same
intent — a logged artifact is one specific kind of thing, not an opaque blob.

The union grows with the project. `BpeTokenizer` joins at W18; `RewardModel` and
`PreferenceData` at W23. Centralizing the union means `_hash_artifact` in
`persistence.py` and any dispatch code stay in sync: the type checker flags
missing cases.

## `Policy`: weights without config

```python
@dataclass(frozen=True)
class Policy:
    model: torch.nn.Module
    value_head: torch.nn.Module | None = field(default=None)
```

`Policy` holds the trained weights. It does **not** hold the architecture config
(vocab size, context length, number of heads). This differs from HuggingFace's
`PreTrainedModel`, which bundles a `PretrainedConfig` with the weights.

The reason: in Demoodle, architecture config lives in the `ArchitectureProtocol`
implementation (`BigramArchitecture`, `MLPArchitecture`, etc.), not in the
artifact. The architecture knows its own shape; the `Policy` does not. This keeps
artifacts lean and makes them loadable by any architecture that understands their
structure — without hardcoding config into the artifact format.

## The `value_head` reservation

`Policy.value_head` is `None` through all training regimes until PPO (W27). PPO
requires a separate scalar value-function head alongside the policy network for
advantage estimation.

Reserving the field now means no architecture, protocol, or artifact format needs
to change when PPO arrives. This mirrors how HuggingFace TRL defines
`AutoModelForCausalLMWithValueHead`: the value head is a natural extension, not a
retrofit.
