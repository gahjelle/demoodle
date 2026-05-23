## 1. Sparkline helper

- [x] 1.1 Implement `_sparkline(losses, width, n_steps) -> str` in `cli.py`: pre-allocate `width` empty braille cells (`⠀`), fill left-to-right using `⠀⣀⣤⣶⣿` (4 levels, normalized to observed min/max), unfilled positions stay `⠀`
- [x] 1.2 Write unit tests for `_sparkline` in `tests/frontends/test_cli.py`: empty losses → all `⠀`; single loss → one filled cell; full run → all filled; midpoint → ~half filled

## 2. Pipeline assembly helpers

- [x] 2.1 Implement `_make_arch(config, vocab_size) -> ArchitectureProtocol` in `cli.py`: `match config.architecture.active` → `BigramArchitecture(vocab_size=vocab_size)`; raise `ValueError` for unknown keys
- [x] 2.2 Implement `_run_pipeline(config, rng, cache_dir) -> dict[str, Artifact]`: phase 1 runs tokenizer stage to get `vocab_size`; phase 2 creates arch and runs the full pipeline; returns all artifacts

## 3. `demoo train` command

- [x] 3.1 Add `@app.command` `train()` to `cli.py`: call `_run_pipeline` with a `Rich.Live` context open during the pretrain stage; update the live display (step header + sparkline) at each log interval
- [x] 3.2 Wire the live sparkline update into the training loop: `make_pretrain_stage` will need a progress callback — add an optional `on_step: Callable[[int, float], None] | None = None` parameter to `make_pretrain_stage` in `training/stages.py`
- [x] 3.3 On cache hit, load `metrics` from result and render completed static sparkline + final loss as summary (same `_sparkline()` fn, all cells filled)
- [x] 3.4 Test `demoo train` invokes without error on the names corpus (integration-style smoke test using `cyclopts` test runner or `subprocess`)

## 4. `demoo call` command

- [x] 4.1 Add `@app.command` `call()` to `cli.py` with params `temperature: float = 1.0`, `n: int = 5`, `prompt: str = "\n"`
- [x] 4.2 Check `_needs_training` before running pipeline: if `True`, print "No trained model found — running train first." and delegate to `train()`; otherwise run pipeline silently
- [x] 4.3 Implement `_generate(arch, policy, tokenizer, rng, temperature, prompt, max_len=100) -> str`: encode prompt, loop calling `arch.call(seq[-context_length:], ...)`, stop on `"\n"` token or `max_len`; return decoded continuation (excluding the prompt)
- [x] 4.4 Loop `n` times, print each result on its own line
- [x] 4.5 Test `_generate` with fixed seed: same seed → same output; `--prompt "Ma"` → output starts with `"Ma"`; stop-token terminates loop

## 5. Verification

- [x] 5.1 `uv run ruff format src/demoodle/frontends/cli.py tests/frontends/`
- [x] 5.2 `uv run ruff check src/demoodle/frontends/cli.py tests/frontends/`
- [x] 5.3 `uv run ty check src/ tests/`
- [x] 5.4 `uv run pytest tests/frontends/`
- [x] 5.5 Manual smoke: `uv run demoo train` completes with decreasing loss; `uv run demoo call` prints plausible names; `uv run demoo train` a second time reports cache hit; `uv run demoo call --temperature 0.1` vs `--temperature 2.0` produces visibly different outputs

## 6. Documentation

- [x] 6.1 Review `CONTEXT.md` — add or update terms for `cli-train`, `cli-call`, `sparkline` if absent
- [x] 6.2 Mark W12 as ✅ in `PLANS.md`
