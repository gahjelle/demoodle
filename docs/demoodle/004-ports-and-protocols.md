# W4 — Protocols & Ports

## Hexagonal architecture

Hexagonal architecture (also called Ports and Adapters, introduced by Alistair
Cockburn) separates the inside of a system from its outside via explicit *ports* —
interfaces that define what the inside needs. Anything outside the core connects
through these ports. Swapping an adapter (a specific implementation) requires no
changes to the core.

In Demoodle:
- `TokenizerProtocol` is the port between the core and any tokenizer
- `ArchitectureProtocol` is the port between the core and any model architecture
- `InspectableProtocol` is the port between the core and anything that can sample
  and optionally explain itself

The training stages, the runner, and the CLI talk to these protocols — never to
specific implementations like `BigramArchitecture` or `CharTokenizer`. This is
what makes swapping tokenizers (W19) or adding architectures (W13, W16) require
only a config change, not a code change.

## Structural subtyping: `Protocol` vs `ABC`

Python's `Protocol` uses **structural subtyping**: a class satisfies a protocol
if it has the right methods and attributes, regardless of whether it inherits from
the protocol. No `class BigramArchitecture(ArchitectureProtocol):` required.

```python
class ArchitectureProtocol(Protocol):
    vocab_size: int
    context_length: int
    def init_state(self, rng: RNG) -> Policy: ...
    def forward(self, policy: Policy, tokens: Seq) -> Output: ...
```

`BigramArchitecture` is a frozen dataclass with `vocab_size`, `context_length`,
`init_state`, and `forward`. It satisfies `ArchitectureProtocol` without declaring
it — duck typing, verified by the type checker at call sites.

The alternative, `ABC` (Abstract Base Class), uses **nominal subtyping**: you
must explicitly inherit from the base class. Nominal subtyping couples
implementations to the interface definition file. If you add a method to the ABC,
all subclasses must be updated. With `Protocol`, adding a method is a breaking
change only if something actually calls it — and the type checker identifies
exactly which implementations are affected.

HuggingFace uses a hybrid: models inherit from `PreTrainedModel` (nominal), but
`AutoModel.from_pretrained` accepts anything with a matching class method
(structural in practice). Protocol-style thinking is why HuggingFace models can
be swapped out with minimal code changes even without strict interface adherence.

## `Stage`: declaring the pipeline contract

```python
@dataclass(frozen=True)
class Stage:
    name: str
    needs: list[str]
    produces: list[str]
    config_hash: str
    run: Callable[[dict[str, Artifact], RNG], dict[str, Artifact]]
```

A `Stage` is a pure declaration: "I consume these named artifacts and produce
these named artifacts." The runner reads `needs` and `produces` to determine
execution order and build cache keys; it never inspects `run` until execution time.

This separation of declaration from implementation is the same pattern that
appears across ML orchestration systems under different names:
- **Airflow operators** declare `upstream_task_ids`; the scheduler resolves order
- **Prefect tasks** declare inputs and outputs via Python type annotations; the
  runner builds the dependency graph from how return values flow between tasks
- **Metaflow steps** declare `self.next()`; the runtime manages traversal
- **Luigi tasks** implement `requires()` and `output()` methods

All share the same insight: separating the *declaration of dependencies* from the
*implementation of work* lets a scheduler reason about the graph independently
of the computation itself.

## `InspectableProtocol.explain()` and the default implementation

```python
class InspectableProtocol(Protocol):
    def explain(self, seq: Seq, policy: Policy) -> dict[str, Any]:
        return {}
```

`explain()` has a default implementation directly in the Protocol — it returns
an empty dict. A class that doesn't define `explain` inherits this behavior and
still satisfies the protocol structurally.

The bigram architecture has no interpretability data to expose. By not implementing
`explain`, it inherits `{}` automatically. The transformer (W17) will override it
to return attention weights per head and layer. Front ends feature-detect
interpretability data by checking whether the result is empty — no branching on
architecture type required.
