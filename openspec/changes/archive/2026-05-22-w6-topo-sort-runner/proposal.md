## Why

The pipeline stages (W4) and artifact cache (W5) are defined but nothing executes them — there is no way to run a graph of stages end-to-end. W6 closes this gap by adding `shell/runner.py`, making the full stage-graph executable with caching and deterministic RNG threading.

## What Changes

- **New**: `demoodle/shell/runner.py` — `run()` function that accepts a list of `Stage`s, initial artifacts, a root `RNG`, and a cache directory, and returns the final artifact dict.
- **New**: Graph validation at call time — raises `ValueError` for duplicate `produces` across stages.
- **New**: Topological sort — stages execute in dependency order; stuck graph (cycle or missing artifact) raises `ValueError` with the names of unsatisfied stages and their missing needs.
- **New**: RNG threading — `rng.split()` called for every stage unconditionally (before cache check) so downstream seeds are cache-state-independent.
- **New**: Cache integration — each stage checks `persistence.load()` before running; saves output on miss.

## Capabilities

### New Capabilities

- `stage-runner`: Execute a list of `Stage`s in topological order with content-addressed caching and deterministic RNG threading.

### Modified Capabilities

<!-- No existing spec-level requirements are changing. -->

## Impact

- **New file**: `src/demoodle/shell/runner.py`
- **New tests**: `tests/shell/test_runner.py`
- **Depends on**: `pipeline-ports` (Stage), `artifact-cache` (persistence)
- **No changes** to existing modules

## Non-goals

- Parallel stage execution
- Progress callbacks or logging hooks
- Persistent DAG state between process runs (cache handles reuse)
- Stage-level retry logic
