# W8 — Character Tokenizer

## The tokenization problem

A language model operates on **tokens** — discrete integer IDs. Raw text is a
sequence of Unicode characters. Tokenization is the mapping between them.

The choice of tokenization strategy has large downstream consequences:
- It determines vocabulary size, which sets the dimensions of every embedding table
  and output projection in the model
- It determines sequence length: the same text tokenized differently can be 5
  tokens or 50, which changes how much context the model sees at once
- It determines what patterns the model can learn: a tokenizer that splits
  "unhappy" into ["un", "happy"] exposes the morphological relationship; one that
  keeps it whole hides it

## The evolution of tokenization

**Character-level** (circa 2015): one token per character. Vocabulary is ~100
(printable ASCII or Unicode subset). Sequences are long — typically 4–5× more
tokens than word-level. The model must learn to compose characters into words,
which requires more training data and compute. Used by early character-RNN work
(Graves 2013, Karpathy 2015 "The Unreasonable Effectiveness of RNNs").

**Word-level** (pre-transformer era): one token per word. Vocabulary is ~100k+.
Unknown words (OOV) become a `<UNK>` token, losing all information about their
content. Used by word2vec (Mikolov et al. 2013) and GloVe embeddings.

**WordPiece** (BERT, 2018): subword tokenizer — words are split into learned
subword units using a greedy algorithm that maximizes training-data likelihood.
Rare words split into pieces; common ones stay whole. Vocabulary ~30k. No OOV
problem. The prefix `##` marks continuation pieces (e.g., "playing" → ["play",
"##ing"]).

**Byte-Pair Encoding / BPE** (GPT-2, 2019): also subword, but built by
iteratively merging the most frequent byte pair in the corpus. No pre-tokenizer
assumptions about word boundaries. GPT-4 uses `cl100k_base` — a BPE tokenizer
with 100,000 tokens. Sequences are 3–5× shorter than character-level.

**SentencePiece** (T5, LLaMA, Mistral): language-agnostic BPE or Unigram
Language Model tokenizer operating on raw Unicode codepoints. Works on any
language without a whitespace-based pre-tokenizer. The default for multilingual
models.

## `CharTokenizer`: the simplest case

```python
char_to_id = {c: i for i, c in enumerate(sorted(set(corpus.text)))}
```

The vocabulary is exactly the set of distinct characters in the corpus, sorted
alphabetically and assigned sequential IDs. For the names corpus this is 27
tokens: the 26 lowercase letters and `\n`.

Advantages over all other schemes:
- No hyperparameters (no vocab size to choose, no merge rules to learn)
- Fully invertible: `decode(encode(s)) == s` for any string built from corpus
  characters — no approximation, no unknown tokens
- Trivially correct: the implementation is two dict lookups

The disadvantages don't matter for the names corpus (short sequences, tiny
vocabulary, CPU-scale training) but will matter for W16 (transformer on
TinyShakespeare), which is why W18 introduces BPE.

## Structural satisfaction

`CharTokenizer` is a frozen dataclass that satisfies `TokenizerProtocol`
structurally — it has `vocab_size`, `encode`, and `decode` but does not inherit
from any base class. This is why `BpeTokenizer` (W18) can be added as a separate
independent type in the `Artifact` union, satisfying the same protocol without
any shared inheritance hierarchy.
