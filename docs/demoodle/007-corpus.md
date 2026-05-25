# W7 — Names Corpus

## What a corpus defines

A **corpus** is the raw text a language model learns from. It defines vocabulary,
style, structure, and statistical patterns. A model can only generate what its
corpus contains — train on names, get name-like output; train on Shakespeare, get
Shakespearean prose.

This coupling between corpus and model capability is one of the most
underappreciated facts about LLMs. GPT-2 was trained on WebText (~40 GB of
Reddit-linked web pages); GPT-3 on a mixture including Common Crawl (~400 GB),
Books (~60 GB), and Wikipedia (~3 GB); LLaMA 3 on ~15 trillion tokens from web,
code, and books. The specific corpus explains which knowledge a model has and
which it lacks — far more than the architecture does.

## The names dataset

`data/names.txt` is a list of human first names, one per line, in lowercase. It
originates from US Social Security Administration public name frequency data and
was popularized as a teaching corpus by Andrej Karpathy's **makemore** project:
a series of progressively more powerful character-level language models (bigram →
MLP → RNN → transformer), all trained on names.

Names are a good teaching corpus because:
- **Small vocabulary**: every name uses only the 26 lowercase letters plus `\n` —
  27 characters total. A character-level tokenizer needs 27 tokens.
- **Clear structure**: names have recognizable shape (initial consonant clusters,
  vowel patterns, suffixes). A trained model produces name-shaped output that's
  easy to evaluate by eye.
- **Fast to train**: the corpus is ~32 KB. A bigram model reaches useful loss in
  seconds on a CPU.
- **Evaluable without tooling**: "Arlena" and "Jorvik" look like names; "Xzqbr"
  does not. No automated metric is needed for a quick sanity check.

## Scale in perspective

```
names.txt           ~32 KB      ~32,000 characters
TinyShakespeare     ~1 MB       ~1,000,000 characters
WebText (GPT-2)     ~40 GB      ~40,000,000,000 characters
LLaMA 3 training    ~15 TB      ~15,000,000,000,000 tokens
```

The gap between a teaching corpus and a production one is roughly seven orders of
magnitude. A model trained on `names.txt` generates plausible name-shaped strings.
A model trained at LLaMA scale can write code, answer questions, and reason across
domains. The difference is almost entirely in corpus scale and diversity — the
architecture changes matter, but they are secondary.
