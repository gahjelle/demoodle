## Context

The codebase has one architecture (`BigramArchitecture`, context_length=1) that uses a V×V weight matrix. The sampling helper `_sample` is a private function inside `bigram.py`, which works for a single architecture but becomes a copy-paste problem the moment a second architecture is added. The `ArchitectureProtocol` already declares `context_length: int` and the training stage already handles `context_length > 1` correctly — the infrastructure for a trigram is essentially already there.

## Goals / Non-Goals

**Goals:**
- Add `TrigramArchitecture` (context_length=2, V×V×V weights) satisfying both `ArchitectureProtocol` and `InspectableProtocol`
- Extract the shared sampling function so all architectures import from one place
- Handle the cold-start case (prompt shorter than context_length) in the generation layer
- Update all docs and config to acknowledge trigram as a first-class selectable architecture

**Non-Goals:**
- Generalised N-gram support (4-gram, 5-gram)
- Automatic architecture selection or dynamic registry
- GPU / performance optimisation for large vocabularies

## Decisions

### 1. Direct V×V×V lookup rather than flattened representation

The trigram weight tensor is `(V, V, V)` — a three-dimensional table indexed by `(t_{n-1}, t_n)`. The alternative (flattening the two context tokens into a V² index, giving a V²×V matrix) would produce the same number of parameters but lose the natural indexing: `weight[a, b]` reads as "given (a, b), predict next" rather than `weight[a*V + b]`.

Direct indexing also makes the educational narrative cleaner: the bigram was `weight[t]`, the trigram is `weight[t_prev, t_cur]`, and the pattern of extending context is immediately visible.

### 2. Cold-start padding lives in `_generate`, not in the architecture

When a sequence is shorter than `context_length`, something must provide the missing context. Two options:
- **In the architecture**: `TrigramArchitecture.forward` detects `len(tokens) < 2` and repeats or synthesises a token
- **In the generation loop** (`_generate` in `cli.py`): pad the sequence before calling `arch.call`

The protocol contract is that architectures receive `context_length` tokens. Architectures should not need special cold-start logic — that couples them to generation concerns. `_generate` already holds the tokenizer and knows what `"\n"` means, so it can pad with `tokenizer.encode("\n")[0]` naturally. The `TrigramArchitecture` then always receives a 2-element context, no special casing needed.

Padding with `"\n"` is the right choice over repeating the first token: `"\n"` is the corpus-level start-of-name token, so `weight["\n", t]` has trained-on signal (every name that starts with character `t` contributes). `weight[t, t]` for most `t` is near-random.

### 3. Shared `sample` in `architectures/sampling.py`

`_sample` was private to `bigram.py`. MLP and transformer will also need sampling. Extracting to `architectures/sampling.py` as the public function `sample` (no underscore) makes it explicitly shared. Importers: `bigram.py`, `trigram.py`, and all future architectures.

The bigram spec's "Sampling helper" requirement is updated to point to the new location and public name.

### 4. Static config, Option A

`ArchitecturesConfig` in `schemas.py` grows a `trigram: TrigramConfig` field alongside `bigram`, `mlp`, and `transformer`. `_make_arch` in `cli.py` gains a `"trigram"` case. Architecture switching is done by editing `active` in `demoodle.toml` manually — no CLI flag, no registry. The three known upcoming architectures (trigram, MLP, transformer) are a list, not a registry problem.

## Risks / Trade-offs

- **V³ memory scaling** → Documented in `docs/architectures/trigram.md`. At V=27 (names corpus) the tensor is 19 K parameters — trivial. At V=1000 (BPE) it is 1 B parameters — impractical. The trigram is char-level only.
- **`_sample` rename breaks `test_bigram.py`** → Intentional. The test import updates alongside the implementation, as it should. There is no re-export alias to maintain.
- **Cold-start only handled in `_generate`** → If any other call site (e.g. a future TUI) calls `arch.call` with a short sequence, it will not automatically get padding. Documented as a responsibility of the caller; architectures trust their context.

## Open Questions

None — all design decisions were resolved in the explore session.
