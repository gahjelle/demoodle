# ADR-0003: Dataset is a flat token sequence; windowing is the training loop's job

**Status:** Accepted

## Context

`build_dataset` takes a `Corpus` and a tokenizer and produces a `Dataset`. The
question was what shape `Dataset` should have — specifically, whether it should
pre-compute `(input, target)` pairs or fixed-length context windows, or whether
it should remain a flat 1D token tensor.

## Decision

`Dataset.tokens` is a flat 1D `torch.long` tensor of token IDs. No windowing,
no pre-computed pairs.

## Reasoning

`build_dataset` takes only `(corpus, tokenizer)` — it has no access to
architecture config. It therefore cannot know the context window size that the
downstream architecture needs. Pre-computing windows would require passing
architecture config into the dataset stage, which would create a dependency
between the dataset and the architecture axes — exactly the coupling the pipeline
design is meant to avoid.

Each architecture's training loop is responsible for slicing the flat sequence
into whatever shape it needs:

- **Bigram**: `inputs = tokens[:-1]`, `targets = tokens[1:]` (single-token pairs)
- **MLP**: sliding windows of size `context_len`
- **Transformer**: chunks of `block_size`

This keeps `build_dataset` architecture-agnostic and the stage graph acyclic.

## Consequences

- `Dataset` has one field: `tokens: torch.Tensor` (1D, dtype `torch.long`)
- The training loop (W11+) is responsible for windowing; `Dataset` carries no
  batch or context-window structure
- Changing context window size requires only a config change and a re-run of the
  training stage — the cached `Dataset` artifact remains valid
- A flat sequence cannot represent variable-length sequences without padding; this
  is acceptable for the demo corpora (names, Shakespeare, code), which are encoded
  as one continuous stream
