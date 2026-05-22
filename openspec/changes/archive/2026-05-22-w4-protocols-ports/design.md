## Context

W2 established the immutable data spine (`core/types.py`). W3 adds the `RNG` threading type (`core/rng.py`). W4 defines the behavioral contracts that sit above the data: what it means to be a tokenizer, an architecture, an inspectable model, or a pipeline stage. These protocols are the swap points the runner (W6) and all concrete implementations (W8, W10, W13, W14) depend on.

The `ports/` package already exists with an empty `__init__.py` and a stub `protocols.py` docstring.

## Goals / Non-Goals

**Goals:**
- Define `TokenizerProtocol`, `ArchitectureProtocol`, `InspectableProtocol` as `typing.Protocol` classes
- Define `Stage` as a frozen dataclass (not a Protocol) with a typed `run` callable
- Make all four names importable from `demoodle.ports`
- Type-check cleanly against dummy implementations

**Non-Goals:**
- Concrete implementations of any protocol
- `FrontendProtocol` — the frontend seam is the W26 shell API
- Runtime `isinstance` checking (`@runtime_checkable` not needed)
- Any training, inference, or persistence logic

## Decisions

### Protocol suffix on all behavioral contracts

`core/types.py` already exports `Tokenizer` (the artifact, a `vocab_size` carrier). Naming the protocol `Tokenizer` in `ports/protocols.py` would shadow it at import sites. The `Protocol` suffix — `TokenizerProtocol`, `ArchitectureProtocol`, `InspectableProtocol` — makes the distinction explicit and consistent. Plain artifact names are always in `core.types`; protocol names always end in `Protocol`.

### `InspectableProtocol.explain` has a default body

`explain()` is optional by design — bigram and MLP return `{}`, transformer returns attention matrices (W15). Python `Protocol` classes can carry default method bodies; implementors that don't override `explain` inherit the default. This is preferable to making `explain` truly optional via `hasattr` checks at every call site.

```python
class InspectableProtocol(Protocol):
    def call(self, seq: Seq, temperature: float) -> int: ...
    def explain(self) -> dict[str, Any]:
        return {}
```

`explain`'s signature is kept minimal now. W15 will revisit it when attention inspection is designed — at that point, passing `Output` or positional context may be needed.

### `ArchitectureProtocol` and `InspectableProtocol` are separate

Architectures implement both: `forward` is the training interface, `call`/`explain` is the inference/inspection interface. Keeping them as separate protocols lets non-architecture objects (e.g., a future sampler wrapper) implement `InspectableProtocol` independently, without being typed as full architectures.

### `Stage.run` carries `RNG`

```python
run: Callable[[dict[str, Artifact], RNG], dict[str, Artifact]]
```

Stages that receive `RNG` explicitly are pure functions of their inputs: same artifacts + same seed → same outputs, with no global state. The runner (W6) calls `stage_rng, rng = rng.split()` before each stage. Stages that don't need randomness simply ignore the argument. The alternative — omitting `RNG` from the signature and relying on closures or global seeding — violates the W3 design principle and makes reproducibility a matter of convention rather than type.

### No `FrontendProtocol`

CLI, TUI, and web frontends are invoked so differently (argparse dispatch, Textual app loop, ASGI server) that a shared `run()` abstraction would be a nominal contract with no structural enforcement. The real seam is the W26 shell API: all frontends consume it and none touch core. Introducing a protocol with no consumer adds noise to a pedagogical codebase.

### `needs` and `produces` are `list[str]`

The project convention is `list` for homogeneous sequences with immutable operations — not `tuple`. `tuple` is for heterogeneous, positional records. The runner's topological sort uses these as node labels in the dependency graph. Strings are used rather than types to keep `Stage` decoupled from the specific artifact classes it handles. Callers treat the lists as read-only by convention.

## Risks / Trade-offs

- **`explain` signature will need to change at W15** → Low risk; it's an internal protocol. When W15 extends it, existing implementations that return `{}` will still satisfy the protocol if the new parameter is optional.
- **`ArchitectureProtocol.init_state` signature is underspecified** → `init_state` needs config and possibly a `RNG` to initialize weights. The exact signature depends on W10. Leaving it with `...` in the protocol is honest; W10 will nail it down.

## Open Questions

None. All material decisions were resolved in the exploration session preceding this change.
