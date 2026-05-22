# ADR-0005: RNG split is unconditional and happens before the cache check

**Status:** Accepted

## Context

The runner threads `RNG` through stages via `.split()`, giving each stage a
deterministic child RNG derived from the parent. The question was whether to split
only when a stage actually runs (cache miss) or unconditionally before every stage,
including cache hits.

## Decision

`stage_rng, rng = rng.split()` executes before the cache lookup for every stage,
regardless of cache state.

## Reasoning

If the split happened only on a cache miss, the seed sequence seen by downstream
stages would depend on how many upstream stages were cached. Adding a new stage
(or warming the cache) mid-graph would shift the seeds of all subsequent stages,
invalidating their cache keys and forcing unnecessary reruns — or worse, silently
producing different results from a run that looks identical.

With unconditional splitting, the seed sequence is a pure function of:
1. The initial RNG seed
2. The topological execution order of stages

Neither of these changes when cache state changes. A stage's `stage_rng` is
therefore stable across runs, and its cache key is stable as long as its inputs
and config are unchanged.

The cost is one `.split()` call per stage per run even on full cache hits. `.split()`
is a SHA-256 hash — negligible.

## Consequences

- `cache_key(stage, inputs, rng)` is called with the post-split `stage_rng`, so
  the seed is baked into the cache key
- Inserting a new stage into the graph shifts the seeds of all stages that execute
  after it in topological order — their cache keys change and they rerun once, then
  re-cache correctly
- This invariant must be preserved by anyone modifying the runner: **never move the
  split inside the cache-miss branch**
