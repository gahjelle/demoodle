## Context

The project follows a Functional Core / Imperative Shell architecture. `core/types.py` is the data spine — every stage in the pipeline produces and consumes these types. The config layer is already complete (Pydantic + configaroo). This change adds the value types that make wiring stages possible.

All types must be immutable (frozen dataclasses). PyTorch is already a dependency.

## Goals / Non-Goals

**Goals:**
- Define all artifact value types as frozen dataclasses
- Keep types minimal: no behavior, no persistence logic, no coupling to architecture config
- Establish the `Artifact` union so the runner (W6) can be typed correctly
- Add architecture explainer docs for the pedagogical audience

**Non-Goals:**
- Protocols / behavioral contracts (W4)
- Persistence / caching (W5)
- Concrete architecture implementations (W10, W13, W14)
- Any training, encoding, or inference logic

## Decisions

### Frozen dataclasses over Pydantic models
Pydantic is already used for config (rich validation, env overrides). The value types are simpler — no validation rules, no coercion. Plain `@dataclass(frozen=True)` is lighter, has no extra dependency, and signals "this is just data." Pydantic is for config; dataclasses are for the core.

### `Seq = torch.Tensor` alias
Token sequences are 1D int64 tensors throughout the pipeline. A type alias documents intent without introducing a new class. Using `torch.Tensor` directly keeps interop with PyTorch ops trivial.

### `Corpus.text: str`
The corpus is raw text. Splitting into words, lines, or characters is a tokenizer/dataset concern — different architectures (names vs Shakespeare) may want different boundaries. Committing to a split in the corpus artifact would force all tokenizers to accept the same segmentation.

### `Dataset.tokens: torch.Tensor` (flat sequence)
`build_dataset` (W9) takes only `(corpus, tokenizer)` — no architecture config. So it cannot know the context window size. The dataset is therefore the encoded token sequence; windowing into `[N, context_len]` batches happens in each architecture's training loop. This keeps `build_dataset` architecture-agnostic and the stage graph acyclic.

### `Policy` holds `nn.Module`, not `state_dict`
A `state_dict`-only Policy is functionally purer but requires reconstructing the module on every forward pass — expensive in generation loops and awkward for inspection. The `Policy` frozen dataclass prevents field reassignment. The semantic contract is write-once: training produces a new `Policy` rather than mutating an existing one. The live module makes forward passes and `explain()` (W15) straightforward.

### `Tokenizer` artifact is a minimal placeholder
`CharTokenizer` (W8) and BPE (W17) will subclass `Tokenizer` or satisfy the `TokenizerProtocol` (W4). For W2, `Tokenizer` holds only `vocab_size: int` — the one field every tokenizer shares. This gives the `Artifact` union a concrete, constructable variant without over-specifying the contract.

### Protocol naming: `TokenizerProtocol`, `ArchitectureProtocol`
Protocols (W4) get the `Protocol` suffix to avoid shadowing the artifact types in `core/types`. This keeps imports unambiguous: `from demoodle.core.types import Tokenizer` is the artifact; `from demoodle.ports.protocols import TokenizerProtocol` is the behavioral contract.

### `Metrics.losses: list[float]`
Plain Python floats are JSON-serializable, easy to display in frontends, and simple to construct in tests. A torch tensor would be idiomatic for GPU training but adds no value at the type definition level. `list[float]` is treated as immutable by convention (consistent with the project's stance on mutable containers).

### Architecture docs format
Files in `docs/architectures/` target readers with Python fluency but limited LLM background. Each file leads with a TLDR block (what it is, when to use it, what makes it interesting) and follows with deeper sections. This mirrors how the demo itself is structured: surface the intuition first, depth on demand.

## Risks / Trade-offs

- **`nn.Module` mutability in Policy** → Mitigation: convention + code review. Training code always creates a new `Policy`; no stage mutates an existing one. Worth documenting explicitly.
- **`Tokenizer` placeholder may drift** → When W8 adds `CharTokenizer`, ensure it subclasses or replaces `Tokenizer` in the `Artifact` union correctly. The union definition is the single place to update.
- **`list[float]` for losses is mutable** → Risk is low (losses are only written once at end of training). If it becomes a problem, switch to `tuple[float, ...]` at that point — it's an internal type, not a public API.

## Open Questions

None. All design decisions were settled during the exploration session.
