## Context

W1–W9 deliver corpus → tokenizer → dataset. The pipeline has no model. W10 adds `BigramArchitecture` — the first concrete implementation of both `ArchitectureProtocol` and `InspectableProtocol`. It also corrects a design flaw discovered during exploration: the original `InspectableProtocol.call(seq, temperature)` signature carried no `policy` argument, which would have forced architectures to hold `Policy` state internally and violated the functional-core invariant. The fix — adding `policy` explicitly to both `call` and `explain` — is a breaking change to the protocol but there are currently no non-stub implementors.

## Goals / Non-Goals

**Goals:**
- `BigramArchitecture` satisfies `ArchitectureProtocol` and `InspectableProtocol`
- `InspectableProtocol` updated: `call(seq, policy, temperature, top_k=None, top_p=None)` and `explain(seq, policy)`
- Sampling: temperature scaling, top-k filtering, top-p nucleus sampling — all in a shared helper, ready for reuse by MLP and transformer
- Deterministic under fixed seed; `forward` is pure; `call` is deterministic given an `RNG`-seeded generator

**Non-Goals:**
- Training loop (W11)
- MLP or transformer architectures
- Meaningful top-k/top-p behaviour at names-corpus vocabulary size (~27)

## Decisions

### BigramModel uses a raw `nn.Parameter`, not `nn.Embedding`

The V×V weight matrix is stored as `nn.Parameter(torch.zeros(vocab_size, vocab_size))` inside a thin `BigramModel(nn.Module)`. Forward: `return self.weight[x]` — a row lookup.

**Why not `nn.Embedding`?** `nn.Embedding` names the concept "embedding", which in later architectures (MLP, transformer) means a smaller learned representation of the vocabulary. Using it here at full V×V size would introduce the term before it means anything, and would make the MLP's embedding (dim < vocab_size) confusing by contrast. A raw parameter is more transparent: the model is literally a lookup table, directly inspectable as a heatmap.

### `BigramArchitecture` is a `frozen=True` dataclass

`BigramArchitecture(vocab_size: int)` — frozen, because it carries no mutable state and `frozen=True` enforces that at runtime. No `Policy` or `nn.Module` is stored on it; every method that needs model state receives `policy` explicitly.

### `call` and `explain` receive `policy` explicitly

`call(seq, policy, temperature, top_k, top_p)` and `explain(seq, policy)`. The architecture is pure config/logic. This mirrors `forward(policy, tokens)` and is the symmetric complement of ADR-0004 ("architecture class is not stored in Policy"). See updated ADR-0004.

### Sampling lives in a module-level helper

```
_sample(logits, temperature, top_k, top_p, generator) -> torch.Tensor
```

Pure function, no class state. Called by `call`; importable by future architectures (MLP, transformer) so they don't re-implement the same top-k/top-p logic.

### `call` uses `seq[-1]` only; `forward` accepts the full sequence

`BigramArchitecture.forward(policy, tokens)` receives the full `Seq` but only indexes `tokens[-1]`. This keeps the calling convention uniform across all architectures — front ends and the pretrain loop never need to know the bigram only uses one token.

### Weight initialisation uses the `RNG` generator

`nn.init.normal_(model.weight, generator=rng.generator())` — no global seed. Determinism flows through the `RNG` value type.

## Risks / Trade-offs

- **Protocol breaking change** → No current implementors beyond stubs; caught before any callers exist.
- **`_sample` helper is internal** → If future architectures need to customise sampling (e.g. beam search), they'll bypass this helper. Acceptable: the helper covers the common case; unusual samplers own their own logic.
- **`forward` silently ignores context beyond last token** → Correct for bigram; a future test that passes a multi-token sequence and expects context-sensitivity would be misleading. Tests should be clear about this being bigram-specific behaviour.

## Open Questions

None. All design decisions resolved during exploration.
