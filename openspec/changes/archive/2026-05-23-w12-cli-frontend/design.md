## Context

W11 delivers the pretrain stage, a trained `Policy`, and `TrainingMetrics` — but no user-facing entry point. W12 wires the day-one pipeline to the `demoo` CLI. The existing `frontends/cli.py` is a stub with only `show_config`; W12 adds `train` and `call` subcommands.

The key structural constraint is that `BigramArchitecture` requires `vocab_size` at construction time, but `vocab_size` is not known until the tokenizer stage runs. This creates a chicken-and-egg problem: `make_pretrain_stage(arch, config)` needs an arch, but arch needs vocab_size from the tokenizer.

## Goals / Non-Goals

**Goals:**
- `demoo train`: run the full day-one pipeline, display a live braille sparkline that also serves as the progress indicator, cache the result
- `demoo call`: run the pipeline (cache or auto-train), generate `n` text continuations using the trained policy
- Architecture dispatch on `config.architecture.active` so W13 (MLP) needs only a new `match` case
- Auto-train with a printed notice when `call` is invoked before any training has been done

**Non-Goals:**
- Interactive sessions, attention visualization, comparison mode (W28/W29)
- Any output format other than plain text to stdout

## Decisions

### 1. Two-phase runner.run to resolve the vocab_size dependency

The CLI calls `runner.run` twice:

**Phase 1** — tokenizer only:
```
runner.run([tok_stage], {"corpus": corpus}, rng, cache_dir)
→ {"corpus": ..., "tokenizer": CharTokenizer(...)}
```
`vocab_size = result["tokenizer"].vocab_size`

**Phase 2** — dataset + pretrain, seeded with the same root RNG:
```
arch = _make_arch(config, vocab_size)
runner.run([tok_stage, dataset_stage, pretrain_stage(arch, ...)],
           {"corpus": corpus}, rng, cache_dir)
→ {"base_policy": Policy(...), "metrics": TrainingMetrics(...), ...}
```

Phase 1 is always a cache hit after the first run (tokenizer stage is deterministic). The root RNG is passed to both calls unchanged — the runner splits internally, so the seed each stage receives is identical whether or not phase 1 was a real run or a cache hit. No caching invariants are broken.

**Alternative considered:** pass `vocab_size=0` as a placeholder for the config hash, then inject the real value at run time. Rejected — silently wrong cache keys would cause subtle bugs if `vocab_size` ever changed.

**Alternative considered:** add "tokenizer" to the pretrain stage's `needs` so it can read `vocab_size` itself. Rejected — the pretrain stage would then need to create the architecture internally, coupling stage logic to architecture construction and making `make_pretrain_stage(arch, config)` impossible.

### 2. Architecture dispatch via `match` on `config.architecture.active`

A single `_make_arch(vocab_size)` helper dispatches on `config.architecture.active` from the module-level config:

```python
def _make_arch(vocab_size):
    match config.architecture.active:
        case "bigram":
            return BigramArchitecture(vocab_size=vocab_size)
        case other:
            msg = f"Unknown architecture {other!r}"
            raise ValueError(msg)
```

`config` is the module-level singleton; no need to pass it explicitly since all CLI helpers use it the same way. W13 and W16 add new `case` branches here — no other CLI change needed. This is the only place in the CLI that knows about concrete architecture types.

### 3. Braille sparkline doubles as progress bar

The display during training is a single `Rich.Live` panel with two lines:

```
Training  step 2350/5000 • loss: 2.891
⣿⣿⣷⣶⣤⣤⣠⣠⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
```

The sparkline strip is pre-allocated to a fixed width (terminal width − margin, min 40). Each character position represents `n_steps / width` steps. Filled positions (left) encode the loss at that window via four braille levels (`⠀⣀⣤⣶⣿`); unfilled positions (right) remain `⠀`. The transition from filled to empty is the progress indicator — no separate bar needed.

Loss values are normalized on the range `[initial_loss, 0]`: the first recorded loss becomes the top of the scale (full bar = `⣿`); zero is the bottom (empty = `⠀`). If loss rises above the initial value (unusual but possible early in training), the bar is clipped to `⣿`. This makes bar height represent absolute progress toward convergence, so the sparkline is comparable across runs and architectures — a half-height bar always means "halfway to zero loss."

**Alternative considered:** normalize to the observed `[min, max]` of the current run. Rejected — this uses the full dynamic range regardless of absolute progress, so a run that barely improves looks identical to one that halves its loss. The `[initial_loss, 0]` scale is more honest for demo purposes.

**Alternative considered:** Rich's `Progress` with a custom `ProgressColumn` subclass. Rejected — `Progress` assumes a monotone 0–100% bar; replacing the bar with a non-monotone sparkline requires fighting the abstraction. `Rich.Live` with a manually updated `Text` renderable is simpler and gives complete control.

### 4. `call`-before-training auto-trains with a notice

`call` runs the full pipeline regardless. It checks for a cached `base_policy` before running by inspecting whether the pipeline would produce a cache miss for `base_policy`. In practice this is just: run the pipeline; if training ran (no cache hit), the live sparkline is shown. If it hit cache, training display is skipped.

Concretely: `call()` checks whether the pretrain cache key exists. If missing, it prints a notice and delegates directly to `train()` — which handles pipeline assembly, the live sparkline, and the static summary on completion. After `train()` returns, `call()` runs `_run_pipeline()` to get the artifacts (now a full cache hit) and proceeds to generation.

```python
if persistence.load(pretrain_key, cache_dir) is None:
    console.print("No trained model found — running train first.")
    train()
artifacts = _run_pipeline(config, rng, cache_dir)
```

This removes the need for a separate `_needs_training` helper — the cache check is a one-liner inline in `call()`.

**Alternative considered:** fail fast with an error and tell the user to run `demoo train`. Rejected — this breaks the demo flow if a presenter runs `call` cold.

### 5. Generation loop

```python
token_ids = tokenizer.encode(prompt)  # prompt defaults to "\n"
for _ in range(max_len):
    rng, step_rng = rng.split()
    seq = torch.tensor(token_ids[-arch.context_length:])
    output = arch.call(seq, policy, step_rng, temperature, top_k, top_p)
    next_id = int(output.sampled_ids.item())
    token_ids.append(next_id)
    if tokenizer.decode([next_id]) == "\n":
        break
result = tokenizer.decode(token_ids[len(tokenizer.encode(prompt)):])
```

Using `token_ids[-arch.context_length:]` makes the loop architecture-agnostic: bigram sees only the last token; MLP sees a context window; transformer sees up to its sequence length.

### 6. No new shell helpers — logic stays in cli.py for now

The pipeline assembly logic and the sparkline renderer live in `cli.py` for W12. W28 will extract a proper shell API when the TUI and web front ends need it. Moving it early would create an abstraction with only one consumer.

## Risks / Trade-offs

- **Two runner.run calls**: Slightly more orchestration code in `cli.py`. Acceptable — phase 1 is always a cache hit after the first run, so runtime cost is negligible.
- **Sparkline normalization**: Loss is normalized to `[initial_loss, 0]` with clipping. If a model never meaningfully converges (loss stays near initial), all bars will be tall throughout — accurate but potentially unexciting. Acceptable for the demo; the names bigram reliably decreases.
- **`context_length` coupling**: The generation loop reads `arch.context_length`. This is a field on `ArchitectureProtocol` added by W11; the CLI depends on W11 being complete.
- **Hardcoded stop token**: The generation loop stops on `"\n"`. This is correct for the names corpus but will need revisiting for Shakespeare or code corpora in later milestones.

## Open Questions

None — all design questions resolved during the explore session that preceded this change.
