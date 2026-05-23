## Context

W10 delivered `BigramArchitecture` with `init_state`, `forward`, and `call` — but no training loop. W11 adds the `pretrain` stage that turns a `Dataset` into a trained `Policy` (and a `TrainingMetrics` artifact). The stage must satisfy W13's hard constraint: MLP trains via the **existing** pretrain stage with only a config change, no stage edits. This means the loop must be generic over architecture from day one.

The existing stage pattern is a factory function that closes over config and returns a `Stage` dataclass. `build_dataset` (`data/stages.py`) establishes this pattern; `pretrain` follows it.

One naming discrepancy exists between the code (`Metrics`) and the domain glossary (`TrainingMetrics`). This design uses the glossary term; renaming the type is in scope for this work item.

## Goals / Non-Goals

**Goals:**
- Generic training loop that works for bigram now and MLP/transformer later without edits
- Config-driven hyperparameters under `[training.pretrain]`, consistent with `[architecture]` and `[tokenizer]` patterns
- Correct cache key: any change to hyperparams, architecture config, dataset, or seed invalidates the cache
- Loss visibly decreases on names

**Non-Goals:**
- MLP or transformer implementations
- SFT, DPO, or PPO loops
- Learning rate schedulers or gradient clipping
- A CLI to invoke training (W12)

## Decisions

### 1. `context_length` on `ArchitectureProtocol`

The training loop must know how wide a context window to extract from the `Dataset`. Each architecture has a natural context length (bigram: 1, MLP: `context_length` from config, transformer: sequence length). Rather than letting each architecture supply its own batch-extraction logic, we expose `context_length: int` as a required field on `ArchitectureProtocol`.

`BigramArchitecture` gets `context_length: int = 1`. The CONTEXT.md entry for `Dataset` confirms this responsibility: "Windowing into batches is the architecture's job."

**Alternative considered:** an `extract_batch(dataset, batch_size, rng)` method on the protocol. Rejected — more surface area, and the windowing algorithm is identical across architectures; only the window width varies.

### 2. Training loop calls `policy.model` directly

The training loop needs batched logits: `policy.model(input_batch) -> logits of shape (B, vocab)`. `ArchitectureProtocol.forward(policy, tokens)` is an inference seam (takes a full sequence, returns one set of logits from the last token). Using it in a training loop would require per-position iteration — correct, but bypasses PyTorch's batching.

Calling `policy.model` directly is appropriate here: the `nn.Module` is the training interface; `arch.forward` is the inference interface. This follows the same reasoning as `init_state` being on the architecture while the `Policy` holds the state.

**Alternative considered:** add a `training_forward(policy, input_batch) -> Output` to the protocol. Rejected — the training loop's needs are captured by `context_length` + a direct `nn.Module` call; a new protocol method adds complexity without enabling new behavior.

### 3. Flat `(context, target)` windowing from Dataset

For any `context_length`, random training pairs are extracted as:

```
offset = random int in [0, len(tokens) - context_length - 1]
input  = tokens[offset : offset + context_length]   # shape (context_length,)
target = tokens[offset + context_length]             # scalar
```

A batch of `batch_size` such pairs is stacked before each forward pass. For bigram (`context_length=1`) this gives `input: (B, 1)` and `target: (B,)` — identical to the `(B, vocab)` cross-entropy shape contract.

**Alternative considered:** sequential (non-random) iteration over the dataset. Rejected — random sampling avoids positional bias and is the standard approach; the `RNG` passed to `stage.run` seeds the batch sampler for reproducibility.

### 4. Config structure: `[training.pretrain]`

Current flat `[training]` becomes `[training.pretrain]` with a dedicated `PretrainConfig(learning_rate, batch_size, n_steps)`. `TrainingConfig` becomes a container: `TrainingConfig(pretrain: PretrainConfig)`. This mirrors `ArchitecturesConfig` and `TokenizersConfig` — no `active` key, since training stages are sequential rather than mutually exclusive.

### 5. `config_hash` composition

The pretrain stage's cache key must encode everything that can change the output:

```
config_hash = sha256(
    pretrain_config fields (lr, batch_size, n_steps)
    + arch.vocab_size
    + arch.context_length
)
```

The architecture's `vocab_size` is not in `PretrainConfig` (it comes from the tokenizer artifact at runtime), but it must be in the hash because it changes the model shape. It is read from `arch.vocab_size` at factory time and baked into `config_hash`.

### 6. Location: `training/stages.py`

Post-training (SFT, DPO, PPO) will each add stages. A `training/` module is the right home — it grows without touching `data/` or `architectures/`. The module is new; no existing file is moved.

## Risks / Trade-offs

- **`policy.model` coupling**: the training loop accesses `policy.model` directly rather than going through a protocol method. If `Policy.model` is ever replaced with something other than an `nn.Module`, the loop breaks. Acceptable — `Policy` is defined to hold `nn.Module`; this is not an abstraction boundary.
- **Transformer incompatibility**: transformer training predicts at every sequence position (target shape `(B, seq_len)` not `(B,)`), which doesn't fit the flat windowing scheme. W16 will need to extend or replace this; the stage is not edited for bigram/MLP but may need a variant for transformer. This is noted in the spec as a known limit.
- **Loss recording granularity**: recording every step produces a large `TrainingMetrics` for long runs. A `log_every` parameter could help but is not added now — the demo uses short runs where step-level detail is fine.

## Open Questions

None — all design questions were resolved during the explore session that preceded this change.
