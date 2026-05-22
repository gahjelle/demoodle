## Context

W8 lands `CharTokenizer` and a `train_tokenizer` stage. W9 adds the next stage in
the pipeline: `build_dataset`, which encodes the corpus text into a flat integer
token sequence. The `Dataset` frozen dataclass already exists in `core/types.py`
and `_hash_artifact` already has a `Dataset` branch in `shell/persistence.py` —
nothing in the type spine or cache needs to change.

The only design question is where the stage factory lives, and how to handle the
corpus boundary signal at the end of the text.

## Goals / Non-Goals

**Goals:**
- `make_build_dataset_stage()` factory producing a `Stage` with
  `needs=["corpus", "tokenizer"]`, `produces=["dataset"]`
- Trailing `\n` normalization so every name-boundary transition appears as a
  training example
- No changes to `Dataset`, `core/types.py`, or `shell/persistence.py`

**Non-Goals:**
- Padding, fixed-length chunking, or batching (W11/W16)
- Config-driven tokenizer selection (W19)
- Any modification to the `Dataset` type

## Decisions

### Module placement: `data/stages.py` (not `data/loaders.py`, not `tokenizers/`)

`build_dataset` bridges two artifact types — `Corpus` and `Tokenizer`. It is a
data-preparation stage, not a data-loading concern (that's `loaders.py`) and not
a tokenizer-implementation concern (that's `tokenizers/`). A dedicated
`data/stages.py` separates the "get raw data" responsibility (`loaders.py`) from
the "wire data into the pipeline" responsibility (`stages.py`).

Alternatives considered: inline in `data/loaders.py` (mixes loading with staging),
in `tokenizers/char.py` (wrong owner — the stage is not specific to `CharTokenizer`
and must work for BPE in W19 without changes), in a top-level `stages/` module
(premature when there is only one stage there).

### Trailing `\n` normalization in the stage (not the loader)

`load_corpus` joins lines with `\n` and produces no trailing newline on the last
name. Without normalization, the final name's last character never generates a
`→ \n` training example. The stage normalizes by appending `\n` if the text does
not already end with one.

This decision belongs in the stage, not the loader. The loader's contract is
faithful representation of the source text. The normalization is a training
semantics decision — it belongs at the point where encoding decisions are made.

Alternatives considered: normalize in `load_corpus` (wrong layer; would affect
all corpus consumers, including corpora where trailing `\n` is not meaningful),
leave as-is (silently incomplete training signal for the last name).

### `dtype=torch.long` (int64) for the token tensor

PyTorch embedding layers and cross-entropy loss expect `torch.long` indices.
Using `torch.long` at creation time avoids silent dtype mismatches downstream.

## Risks / Trade-offs

- **`tokenizer` artifact is typed as `Artifact` in the stage `run` function** —
  the runner passes a `dict[str, Artifact]`, so the stage must narrow the type
  with `assert isinstance(tokenizer, CharTokenizer)` or use a structural check.
  This is a known pattern friction until W19 adds config-driven selection; for
  now a direct `isinstance` check is acceptable.
- **Corpus must fit in memory** — the full encoded token list is materialized
  before constructing the tensor. Fine for all planned demo corpora (< 2 MB).

## Migration Plan

No migration needed — no existing `Dataset` artifacts are cached.

## Open Questions

None.
