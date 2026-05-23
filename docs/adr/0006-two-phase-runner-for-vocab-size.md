# ADR-0006: Two-phase `runner.run` to resolve `vocab_size` before constructing the architecture

**Status:** Accepted

## Context

`BigramArchitecture` (and all future architectures) require `vocab_size` at construction time. But `vocab_size` is not known until the tokenizer stage has run — it is a property of the corpus, not of the config. This creates a dependency ordering problem: `make_pretrain_stage(arch, config)` needs a fully constructed arch, but arch construction needs an artifact that only exists after the pipeline has partially run.

## Decision

The CLI runs `runner.run` twice with the same root RNG:

**Phase 1** — tokenizer stage only:
```python
phase1 = runner.run([tok_stage], {"corpus": corpus}, rng, cache_dir)
vocab_size = phase1["tokenizer"].vocab_size
```

**Phase 2** — full pipeline:
```python
arch = _make_arch(vocab_size)
runner.run([tok_stage, dataset_stage, pretrain_stage], {"corpus": corpus}, rng, cache_dir)
```

Phase 1 is always a cache hit after the first run (the tokenizer stage is deterministic and fast). Passing the same root RNG to both calls is safe: the runner splits internally, so each stage receives the same seed regardless of whether phase 1 was a real run or a cache hit.

## Alternatives considered

**Pass `vocab_size=0` as a placeholder for the config hash, inject the real value at runtime.** Rejected — silently wrong cache keys if `vocab_size` ever changes; subtle cache invalidation bugs.

**Add `tokenizer` to `pretrain_stage.needs` so the stage reads `vocab_size` itself.** Rejected — the pretrain stage would then be responsible for architecture construction, coupling stage logic to arch types and making `make_pretrain_stage(arch, config)` impossible to call from outside the pipeline.

**Single-phase pipeline with a lazy arch wrapper.** Rejected — adds indirection for no gain; the two-phase call is explicit and easy to follow.

## Consequences

- `_run_pipeline` in `cli.py` always makes two `runner.run` calls; the first is a no-op after the initial run.
- Any future CLI front end (TUI, web) must replicate this two-phase pattern or extract it into a shared helper.
- New architectures only need a new `case` branch in `_make_arch` — no change to the pipeline assembly.
