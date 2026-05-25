# W9 — build_dataset Stage

## The next-token prediction objective

Language models are trained to answer one question: **given a sequence of tokens,
what is the next token?** This is called the **autoregressive** language modeling
objective.

The central insight of Bengio et al.'s 2003 paper *A Neural Probabilistic Language
Model*: represent words as dense vectors, condition the next-word distribution on
a fixed window of preceding words, and train end-to-end on raw text. The objective
is **self-supervised** — the label is always the next token in the corpus, so no
human annotation is needed. Raw text is its own supervised signal.

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), LLaMA (2023), and virtually every
modern autoregressive language model trains on this same objective. The scale
changed by seven orders of magnitude; the task formulation did not.

## Building the dataset

```python
ids = tokenizer.encode(text)
tokens = torch.tensor(ids, dtype=torch.long)
```

The entire corpus becomes a single flat sequence of token IDs. There is no
sentence-level splitting, no padding, no special tokens — just a long integer
tensor. This is also how GPT-2 prepares its training data (with an `<|endoftext|>`
token between documents, but otherwise flat).

The `build_dataset` stage stores this flat sequence as a `Dataset` artifact. The
training loop slices windows from it at runtime:

```
tokens:   [a, b, c, d, e, f, ...]
input:    [a, b, c]   ← context_length = 3, starting at offset 0
target:   d           ← the next token
```

The target is the token immediately after the input window — shifted by exactly
one position. Every position in the corpus is simultaneously a context for
predicting the next token and a label for the preceding context. One 32 KB corpus
provides ~32,000 training examples with zero additional annotation.

## Context length

The `context_length` parameter determines how many preceding tokens the model sees
when predicting the next one:

- **Bigram (W10)**: `context_length = 1`. The model sees only the immediately
  preceding token. All longer-range structure is invisible.
- **MLP (W13)**: `context_length = N` (a small fixed window). The model sees
  a flat concatenation of the N preceding token embeddings — no positional
  distinction without positional encoding.
- **Transformer (W16)**: `context_length = N` (hundreds to thousands). The
  attention mechanism allows the model to selectively attend to any subset of the
  preceding tokens.

The `build_dataset` stage is context-length-agnostic — it stores the flat token
sequence and lets the training loop extract the appropriate window per step. No
dataset rebuild is needed when switching architectures.

## Why `torch.long`

Token IDs are integers used as indices into embedding tables and weight matrices.
They must be `int64` (`torch.long`) rather than `int32` or `float32` because
PyTorch's embedding and indexing operations require 64-bit integer indices. Using
the wrong dtype here raises an error deep inside the model forward pass — making
the dtype explicit at construction prevents it.
