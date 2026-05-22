# ADR-0001: Use `cast()` for artifact extraction rather than a monolithic TypedDict

**Status:** Accepted

## Context

Stage `run` functions receive a `dict[str, Artifact]` and extract specific artifacts
by key (e.g. `artifacts["corpus"]`). Because the dict value type is the full
`Artifact` union, the type checker cannot narrow the result — every extraction
returns `Artifact`, not the specific type the stage actually needs.

Two approaches were considered:

**Option A — `cast()`**: Keep `dict[str, Artifact]` and use `cast("Corpus", artifacts["corpus"])` at each extraction site. `cast` is a no-op at runtime; it only informs the type checker.

**Option B — Monolithic TypedDict**: Define one `ArtifactDict(TypedDict, total=False)` with a field per artifact key (`corpus: Corpus`, `tokenizer: TokenizerProtocol`, `dataset: Dataset`, …). The type checker then knows the type of each key without any cast.

## Decision

Use `cast()` (Option A).

## Reasoning

Option B requires every artifact key name to be a declared field in `ArtifactDict`. A plugin stage that introduces a new key (e.g. `"my_policy"`) would need to modify `core/types.py` to register it — breaking the W31 "plug-and-play" guarantee that a new stage can be wired in with zero edits to core.

With `cast()`, a plugin is fully self-contained: it declares its `needs`/`produces` key names and casts its own extractions. Core is untouched.

The cost is a small amount of trust-me annotation at each extraction site, which is acceptable given that `needs`/`produces` already encode the same contract at runtime.

## Consequences

- Every `artifacts["<key>"]` extraction in a stage `run` uses `cast("<Type>", artifacts["<key>"])`
- The `Artifact` union in `core/types.py` grows when new concrete artifact types are added, but key names are not registered anywhere in core
- Plugins can introduce arbitrary key names without touching core
