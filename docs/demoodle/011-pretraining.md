# W11 — Pretrain Stage

## What "pretraining" means

Until 2018, most NLP models were trained directly on task-specific labeled data: a
sentiment classifier on movie reviews, a translation model on parallel corpora.
Each task needed its own dataset and its own model trained from scratch.

The **pretraining paradigm** changed this. Train a large model on raw text using
a self-supervised objective (next-token prediction for GPT; masked token prediction
for BERT). The model builds internal representations of language — grammar, facts,
common reasoning patterns — without any task labels. Then fine-tune on a small
labeled dataset for a specific task.

The pretraining step is expensive; the fine-tuning step is fast. One pretrained
model can be fine-tuned for many tasks. This is why the term "foundation model"
emerged: pretraining builds the foundation; fine-tuning adapts it.

Demoodle's `pretrain` stage is a miniature version of this first step: train a
model on raw text via next-token prediction and save the result as `base_policy`.

## The training loop

```python
for step in range(config.n_steps):
    offsets = torch.randint(n - context_len, (config.batch_size,), generator=generator)
    inputs = torch.stack([tokens[o : o + context_len] for o in offsets])
    targets = tokens[offsets + context_len]

    logits = policy.model(inputs)
    loss = F.cross_entropy(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Each iteration:
1. **Sample a batch**: randomly select `batch_size` starting positions in the token
   sequence. Random sampling (rather than sequential) reduces variance in gradient
   estimates — the gradient from a random mini-batch is an unbiased estimate of
   the full-corpus gradient. Sequential sampling can introduce harmful correlations
   when similar tokens cluster together.
2. **Forward pass**: run the model on each context window to get logits.
3. **Loss**: cross-entropy between predicted logits and actual next tokens.
4. **Backward**: `loss.backward()` computes gradients via automatic differentiation
   through the entire computation graph.
5. **Update**: the optimizer adjusts each weight to reduce loss.

## Cross-entropy loss

Cross-entropy measures how surprised the model is by the actual next token:

```
loss = -log(P(correct_token))
```

If the model assigns 90% probability to the correct token, loss is `-log(0.9) ≈
0.1` — low surprise, good prediction. If it assigns 1% probability, loss is
`-log(0.01) ≈ 4.6` — high surprise, poor prediction.

For a uniform distribution over V tokens, the baseline loss is `log(V)`. For the
names vocabulary (V = 27), the random-guess baseline is `log(27) ≈ 3.3`. Training
succeeds when loss drops meaningfully below this baseline — the model has learned
something about which tokens follow which.

`F.cross_entropy(logits, targets)` applies log-softmax to logits and computes
negative log-likelihood in a single numerically stable operation. The numerical
stability matters: naive softmax followed by log can produce `log(0)` for very
negative logits; the fused operation uses the log-sum-exp trick to avoid this.

## Adam optimizer

Adam (Adaptive Moment Estimation, Kingma & Ba 2014) is the default optimizer for
most deep learning. It maintains per-parameter moving averages of gradients (first
moment, `m`) and squared gradients (second moment, `v`), and adapts the effective
learning rate per parameter:

```
m = β₁·m + (1-β₁)·grad
v = β₂·v + (1-β₂)·grad²
param -= lr · m̂ / (√v̂ + ε)
```

Parameters that receive large, noisy gradients get a smaller effective step.
Parameters with small, consistent gradients get a larger one. This adaptive
behavior makes Adam converge faster than vanilla SGD on most language modeling
tasks, and it became the standard optimizer for transformers.

## `TrainingMetrics` as an artifact

The training loop produces `TrainingMetrics(losses=[...])` alongside `base_policy`.
Recording the loss curve as a first-class artifact means:
- The runner caches it alongside the model — no separate logging infrastructure
- The CLI can display it after training without re-running anything
- Future evaluation stages can compare loss curves across configurations

Production ML platforms — MLflow, Weights & Biases, Comet — record metrics as
structured time-series attached to a "run". `TrainingMetrics` is Demoodle's
minimal version: a flat list of per-step losses, produced by training and consumed
by the front end.
