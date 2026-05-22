## Context

W7 landed a `Corpus` artifact and `load_corpus()`. W8 introduces the first concrete
tokenizer — `CharTokenizer` — and the stage that trains it from a corpus. This
unblocks W9 (dataset encoding) and W10 (bigram model).

The existing `Tokenizer` placeholder in `core/types.py` (a frozen dataclass with
only `vocab_size`) was always a transitional stub — W8 removes it and adds
`CharTokenizer` to the `Artifact` union directly.

## Goals / Non-Goals

**Goals:**
- `CharTokenizer` frozen dataclass satisfying `TokenizerProtocol` structurally
- `make_train_tokenizer_stage()` factory producing a runnable `Stage`
- `CharTokenizer` in the `Artifact` union and hashed by `_hash_artifact`
- Remove the `Tokenizer` placeholder from `core/types.py`

**Non-Goals:**
- Config-driven tokenizer selection (W19)
- BPE or any sub-word tokenizer (W18)
- Special boundary tokens (`<BOS>`, `<EOS>`)

## Decisions

### Module placement: `tokenizers/` at the same level as `architectures/`

`CharTokenizer` is a domain-implementation layer, not a core value type and not
shell I/O. It sits alongside `architectures/`, which holds implementations of
`ArchitectureProtocol`. Defining `CharTokenizer` inside `core/types.py` would mix
the type spine with implementation detail.

Alternatives considered: inline in `core/types.py` (pollutes the type spine with
implementation), inside `shell/` (wrong — no I/O or orchestration).

### `\n` as natural word boundary — no special tokens

The names corpus joins lines with `\n`; `\n` enters the vocabulary as a regular
character. The model learns that `\n` marks a name boundary from data. Special
`<BOS>`/`<EOS>` tokens are unnecessary complexity at this stage and would require
`CharTokenizer` to handle escape logic.

Alternatives considered: `|` separator (arbitrary, hides the newline semantics),
`<EOS>` token (useful for explicit stopping but not needed until generation is
wired up in W12).

### `dict[str, int]` in a frozen dataclass

`frozen=True` prevents field reassignment; the dict is constructed once and never
mutated. `hash(tokenizer)` is never called directly — `_hash_artifact` provides
the canonical content hash. This matches how `Policy` stores a `torch.nn.Module`
(also unhashable) under a frozen dataclass.

Alternatives considered: `chars: tuple[str, ...]` with indices as ids (clean and
hashable, but departs from the `{char:id}` framing in PLANS.md); storing both
`char_to_id` and `id_to_char` (redundant for a demo-scale vocab).

### Stage as a factory function, not a module-level singleton

`make_train_tokenizer_stage(config_hash)` returns a `Stage` instance. This matches
the established pattern (runner consumes a list of `Stage` objects) and makes the
config hash explicit at construction time.

## Risks / Trade-offs

- **`id_to_char` rebuilt on every `decode` call** → negligible for demo-scale
  vocabs (< 100 chars); acceptable for now, revisit if profiling shows otherwise.
- **`Tokenizer` placeholder removal** → was a no-op; the placeholder had already
  been removed before W8 landed.

## Implementation notes (post-implementation)

### Circular import resolution

The original plan used a `TYPE_CHECKING`-guarded import of `CharTokenizer` in
`core/types.py`. This broke when any code accessed `Artifact.__value__` at runtime
(the lazy `type` alias evaluates in the module namespace, where `CharTokenizer`
wasn't present).

The fix was **Option A — remove the isinstance check from the stage `run`
function**. `char.py` had imported `Corpus` only to validate its input at runtime;
without that check, `Corpus` becomes a type-annotation-only dependency and moves
under `TYPE_CHECKING`. This breaks the cycle entirely, and `core/types.py` can
import `CharTokenizer` at the top level without any ordering tricks.

This is consistent with the Functional Core / Imperative Shell principle: stages
trust their inputs; the runner and type system enforce correctness at the boundary.

## Migration Plan

1. ~~Remove `Tokenizer` placeholder class from `core/types.py`~~ (already done)
2. Add `CharTokenizer` import and update `Artifact` union in `core/types.py`
3. Add `CharTokenizer` match arm to `_hash_artifact` in `shell/persistence.py`

No data migration needed — no serialised `Tokenizer` artifacts exist in any cache.

## Open Questions

None.
