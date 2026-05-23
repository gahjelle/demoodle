## Why

The pipeline has corpus, tokenizer, and dataset artifacts but no model yet. W10 introduces the first trainable architecture — a learned bigram — completing the data spine needed by the pretrain stage (W11) and making the day-one slice demonstrable.

## What Changes

- **New file** `src/demoodle/architectures/bigram.py`: `BigramModel` (thin `nn.Module` wrapping a V×V `nn.Parameter`) and `BigramArchitecture` satisfying both `ArchitectureProtocol` and `InspectableProtocol`
- **BREAKING: `InspectableProtocol` signature change** — `call` and `explain` both gain an explicit `policy` parameter: `call(seq, policy, temperature, top_k=None, top_p=None)` and `explain(seq, policy)`. Architectures are stateless config/logic; all model state lives in `Policy`. This mirrors the existing `forward(policy, tokens)` design and prevents architectures from holding a `Policy` reference internally.
- `src/demoodle/ports/protocols.py`: update `InspectableProtocol` method signatures

## Capabilities

### New Capabilities

- `bigram-architecture`: The V×V weight matrix model — `init_state`, `forward`, `call` (with sampling), and `explain` (returns `{}`). Covers the `BigramModel` nn.Module, `BigramArchitecture` class, top-k/top-p sampling helpers, and determinism requirements.

### Modified Capabilities

- `pipeline-ports`: `InspectableProtocol.call` gains `policy` and optional `top_k`/`top_p` parameters; `explain` gains `policy`. The stateless-architecture invariant is now a protocol-level requirement.

## Impact

- `ports/protocols.py` — method signatures updated (breaking for any existing `InspectableProtocol` implementors; currently none beyond stubs)
- `architectures/bigram.py` — new file
- `openspec/specs/pipeline-ports/spec.md` — requirement and scenarios updated
- No runner, stage, or persistence changes required

## Non-goals

- Training loop (W11)
- Any architecture beyond bigram (MLP is W13, transformer is W16)
- Top-k/top-p producing meaningfully different output at vocab_size ~27 — they are wired in for interface completeness, not demo value
