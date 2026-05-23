## 1. Update InspectableProtocol

- [x] 1.1 Update `InspectableProtocol.call` signature in `ports/protocols.py`: add `policy: Policy` as second positional arg, add `top_k: int | None = None` and `top_p: float | None = None` after `temperature`
- [x] 1.2 Update `InspectableProtocol.explain` signature: add `policy: Policy` as second positional arg
- [x] 1.3 Update TYPE_CHECKING imports in `ports/protocols.py` to include `Policy` in the `call` and `explain` type hints

## 2. Implement BigramModel and BigramArchitecture

- [x] 2.1 Create `src/demoodle/architectures/bigram.py` with `BigramModel(nn.Module)`: single `nn.Parameter` weight of shape `(vocab_size, vocab_size)`, `forward(x) -> weight[x]`
- [x] 2.2 Add `_sample(logits, temperature, top_k, top_p, generator) -> torch.Tensor` module-level helper: apply temperature, optionally zero out non-top-k logits, optionally apply nucleus (top-p) filter, sample with `torch.multinomial`
- [x] 2.3 Add `BigramArchitecture` dataclass: `vocab_size: int`, implement `init_state(rng) -> Policy` (initialise weights with `rng.generator()`, return `Policy(model=BigramModel(...))`)
- [x] 2.4 Implement `forward(policy, tokens) -> Output`: `policy.model(tokens[-1])`, return `Output(logits=logits)`
- [x] 2.5 Implement `call(seq, policy, temperature, top_k=None, top_p=None) -> Output`: run forward, call `_sample`, return `Output(logits=logits, sampled_ids=sampled)`
- [x] 2.6 Implement `explain(seq, policy) -> dict`: return `{}`

## 3. Tests

- [x] 3.1 Test `BigramModel`: correct output shape, row-lookup behaviour, single parameter group
- [x] 3.2 Test `init_state`: returns `Policy` with `BigramModel`, weight shape `(vocab_size, vocab_size)`, deterministic under same seed
- [x] 3.3 Test `forward`: correct logit shape, identical output for different-length sequences ending in same token, `sampled_ids is None`
- [x] 3.4 Test `call`: `sampled_ids` in `[0, vocab_size)`, temperature affects distribution sharpness (many draws), `top_k=1` always returns argmax, `top_p=0.0` always returns argmax
- [x] 3.5 Test `explain`: returns `{}`
- [x] 3.6 Test updated `InspectableProtocol`: dummy class with new `call` signature satisfies protocol; `explain` default returns `{}`

## 4. Verification

- [x] 4.1 `uv run ruff format src/ tests/`
- [x] 4.2 `uv run ruff check src/ tests/`
- [x] 4.3 `uv run ty check src/ tests/`
- [x] 4.4 `uv run pytest`

## 5. Documentation

- [x] 5.1 Mark W10 done (✅) in `PLANS.md`
- [x] 5.2 Review `CONTEXT.md` — verify `InspectableProtocol` glossary entry matches implemented signatures (already updated during exploration, confirm it matches code)
