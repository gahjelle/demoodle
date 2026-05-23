## Why

The day-one slice has a trained bigram but no way to invoke it from the command line. W12 closes that gap: a `train` command that runs the full pipeline with a live loss display, and a `call` command that generates text from a trained policy — completing the day-one "Definition of Done."

## What Changes

- Add `demoo train` — assembles the day-one pipeline (corpus → tokenizer → dataset → pretrain), runs it via the stage runner, and displays a live braille sparkline that doubles as a progress bar during training.
- Add `demoo call [--temperature] [--n] [--prompt]` — runs the full pipeline (hitting cache when already trained), then generates `n` continuations from the trained policy using the active architecture.
- If `call` is invoked before any training has been done, it prints a brief notice and runs training first before generating.
- Architecture selection is dispatched on `config.architecture.active`, so future architectures (MLP, transformer) require only a new `match` case — no structural CLI change.

## Capabilities

### New Capabilities
- `cli-train`: The `demoo train` command. Assembles and runs the full day-one pipeline. Displays a live braille sparkline + current step + current loss during training; the sparkline fills left-to-right and doubles as a progress indicator.
- `cli-call`: The `demoo call` command. Runs the full pipeline (cache or auto-train), then generates continuations by repeatedly invoking `arch.call()`. Stops each continuation on a newline token (or `max_len=100`). Accepts `--temperature`, `--n`, and `--prompt` (defaults to `"\n"`).

### Modified Capabilities

## Impact

- `src/demoodle/frontends/cli.py` — add `train` and `call` commands
- `tests/frontends/test_cli.py` — new test file
- Depends on W11: `training/stages.py`, `PretrainConfig`, `TrainingMetrics`, `context_length` on `ArchitectureProtocol`

## Non-goals

- Interactive REPL or session management (W29 TUI)
- Attention visualization or inspection output (W17, W28)
- Comparison across checkpoints or architectures (W28)
- Any front-end beyond text output to stdout
