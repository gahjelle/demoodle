## Why

The pipeline has a corpus but no way to turn text into token ids. `CharTokenizer`
is the simplest possible tokenizer — one token per character — and is the first
concrete implementation of `TokenizerProtocol`. It unlocks W9 (dataset) and W10
(bigram) and establishes the tokenizer seam that BPE (W18) will later slot into
without any downstream changes.

## What Changes

- Add `src/demoodle/tokenizers/` subpackage with `char.py`
  - `CharTokenizer`: frozen dataclass, `char_to_id: dict[str, int]`, satisfies
    `TokenizerProtocol` structurally (no inheritance); `vocab_size` derived as
    `len(char_to_id)`
  - `make_train_tokenizer_stage()`: stage factory producing a `Stage` with
    `needs=["corpus"]`, `produces=["tokenizer"]`; builds the char→id mapping from
    the sorted unique characters in the corpus text; `\n` is included as a
    natural name-boundary character — no special tokens
- Add `CharTokenizer` to the `Artifact` union in `core/types.py`
- Add `CharTokenizer` match arm to `_hash_artifact` in `shell/persistence.py`

## Capabilities

### New Capabilities

- `char-tokenizer`: Encode text to a list of integer token ids and decode back,
  one character per token; vocabulary built from a corpus; round-trips arbitrary
  strings drawn from the training alphabet

### Modified Capabilities

- `artifact-types`: `CharTokenizer` joins the `Artifact` union alongside
  `Corpus`, `Dataset`, `Policy`, `Metrics`
- `artifact-cache`: `_hash_artifact` gains a `CharTokenizer` branch so tokenizer
  artifacts are content-addressed correctly

## Design decisions

- **Module placement**: `tokenizers/` sits at the same level as `architectures/` —
  neither core nor shell, a domain-implementation layer that satisfies a protocol.
  `CharTokenizer` is defined there (not in `core/types.py`) to keep the type spine
  free of implementation detail.
- **`\n` as word boundary**: The names corpus joins lines with `\n`; `\n` enters
  the vocabulary as a regular character. No `<BOS>`/`<EOS>` tokens — the model
  learns the boundary signal from data, keeping `CharTokenizer` free of
  special-token logic.
- **`dict` in a frozen dataclass**: `frozen=True` prevents field reassignment; the
  dict is constructed once and never mutated. `hash(tokenizer)` is never called
  directly — `_hash_artifact` provides the canonical hash.

## Impact

- New: `src/demoodle/tokenizers/__init__.py`, `src/demoodle/tokenizers/char.py`
- Modified: `src/demoodle/core/types.py`, `src/demoodle/shell/persistence.py`
- Downstream W9+ stages consume `artifacts["tokenizer"]` — no runner edits needed

## Non-goals

- BPE or any learned sub-word tokenizer (W18)
- Special boundary tokens (`<BOS>`, `<EOS>`, `<PAD>`)
- Config-driven tokenizer selection (W19 wires that up)
- A `train_tokenizer` function outside the `Stage` abstraction
