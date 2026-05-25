# W10 — Learned-Bigram Architecture

## N-gram language models

Before neural networks, language modeling was dominated by **n-gram models**: count
how often each sequence of n tokens appears in the corpus, and use those counts to
estimate the probability of the next token given the preceding n-1.

A **bigram** (n=2) estimates `P(token_t | token_{t-1})`: the probability of each
token given only the one immediately before it. Shannon (1948) used bigrams to
generate English-like text in his foundational information theory paper — Demoodle's
bigram is essentially Shannon's model, parameterized by a weight matrix instead
of raw counts.

N-gram models have an acute limitation: they require storing counts for every
observed sequence, which grows exponentially with n. Most possible n-grams are
never seen in training — counts are sparse and estimates are unreliable for rare
sequences. Bengio et al. (2003) replaced the count table with a neural network:
the same inputs, a differentiable parameterization, and generalization to unseen
sequences.

## The V×V weight matrix

```python
self.weight = nn.Parameter(torch.zeros(vocab_size, vocab_size))
```

Row `i` of the weight matrix is the **logit vector** for token `i`: unnormalized
log-probabilities of each possible next token, given that the current token is `i`.
Applying softmax to row `i` gives the next-token probability distribution.

Forward is a single row lookup:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.weight[x]
```

`x` is a token ID (an integer scalar). `self.weight[x]` retrieves row `x` — a 1D
vector of `vocab_size` logits. This indexing operation is differentiable: the
gradient of the loss flows back to exactly the row that was selected. All other
rows receive zero gradient for this example.

At initialization, all logits are zero (uniform distribution after softmax).
Gradient descent updates rows to increase the probability of tokens that actually
follow each given token in the training data. For the names corpus, the `\n` row
will gradually learn to assign high probability to letters that begin names.

## Sampling: temperature, top-k, top-p

The model produces raw **logits**. To generate text, those logits are converted to
a probability distribution and sampled. Three controls shape the distribution:

**Temperature** divides the logits before softmax:
```python
scaled = logits / temperature
```
Temperature < 1 sharpens the distribution — the model becomes more confident,
more likely to pick its top choice. Temperature → 0 converges to argmax. Temperature
> 1 flattens the distribution toward uniform, increasing diversity at the cost of
coherence. All major LLM APIs (OpenAI, Anthropic, Google) expose temperature as a
generation parameter.

**Top-k** restricts sampling to the k most likely tokens, masking all others to
`-inf` before softmax:
```python
_, top_indices = torch.topk(scaled, k)
mask = torch.full_like(scaled, float("-inf"))
mask.scatter_(0, top_indices, scaled[top_indices])
```
Top-k was introduced in Fan et al. (2018) and popularized by the GPT-2 release as
a way to avoid sampling from the long tail of low-probability tokens.

**Top-p (nucleus sampling)** restricts to the smallest set of tokens whose
cumulative probability exceeds `p`:
```python
cumulative = torch.cumsum(sorted_probs, dim=-1)
to_remove = (cumulative - sorted_probs) > top_p
```
Unlike top-k, the nucleus size adapts to the distribution. A confident model
(peaked distribution) has a small nucleus; an uncertain model (flat distribution)
has a larger one. Introduced by Holtzman et al. (2019), "The Curious Case of
Neural Text Degeneration", this paper is also the source of evidence that purely
greedy or beam-search decoding produces repetitive, degenerate text.

## `forward` vs `call`: training vs inference

`forward(policy, tokens) -> Output` is the training path: compute logits, return
them. No sampling, no randomness, no side effects.

`call(seq, policy, rng, temperature, top_k, top_p) -> Output` is the inference
path: compute logits, sample a next token, return both. It wraps `forward` and
adds the sampling stack.

This split mirrors HuggingFace Transformers: `model.forward()` returns
`BaseModelOutput` with logits; `model.generate()` wraps it with a configurable
sampling loop (greedy, beam search, top-k, top-p, temperature). Keeping training
and inference separate ensures no sampling code appears in the gradient path, and
no gradient machinery appears in inference.

## Stateless architecture, explicit Policy

`BigramArchitecture` is a frozen dataclass. It holds `vocab_size` and
`context_length` — configuration — but never holds a `Policy` reference.

Every method that needs weights receives the `Policy` explicitly:
```python
def forward(self, policy: Policy, tokens: Seq) -> Output:
    logits: torch.Tensor = policy.model(tokens[-1])
    return Output(logits=logits)
```

This pattern comes from JAX/Flax: in JAX, models are pure functions of explicit
parameter arrays. HuggingFace bundles weights into the model object (`self.weight`
is a `Parameter` on the `nn.Module`). Demoodle takes the JAX stance because it
makes `Policy` an inspectable, cacheable artifact — you can hash it, save it, load
it — rather than a live mutable object.
