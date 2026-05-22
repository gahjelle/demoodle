## Context

`Stage` (W4) and `persistence` (W5) are fully implemented. A `Stage` declares `needs`, `produces`, `config_hash`, and `run(inputs, rng) → outputs`. The cache exposes `cache_key(stage, inputs, rng)`, `save`, and `load`. Nothing yet wires stages together into an executable graph.

The runner lives in `shell/` — the imperative shell — and never leaks into `core/` or `ports/`.

## Goals / Non-Goals

**Goals:**
- Execute an arbitrary list of `Stage`s in correct dependency order
- Validate the graph (no duplicate produces) at call time, before any execution
- Detect cycles / unsatisfiable dependencies with an actionable error message
- Thread `RNG` deterministically through every stage
- Use the persistence cache transparently

**Non-Goals:**
- Parallel execution
- Progress callbacks
- Persistent DAG metadata between process runs
- Stage retry or error recovery

## Decisions

### 1. Single `run()` function, not a class

**Decision**: `run(stages, initial_artifacts, rng, cache_dir) → dict[str, Artifact]`

**Rationale**: A function is a pure transformation — no mutable state to reason about, nothing to serialize. A `Runner` class would add lifecycle concerns (construct, configure, execute) with no benefit at this scale. The functional core principle applies here.

**Alternative considered**: `Runner` class with `.add_stage()` / `.execute()` — rejected, unnecessary statefulness.

---

### 2. Topological sort via iterated "ready-set" extraction

**Decision**: Repeatedly collect all stages whose `needs` are satisfied, sort by name for determinism, advance available set, repeat until empty or stuck.

```
available = frozenset(initial_artifact_names)
remaining = list(stages)
order: list[Stage] = []

while remaining:
    ready = sorted([s for s in remaining if all(n in available for n in s.needs)],
                   key=lambda s: s.name)
    if not ready:
        raise ValueError(unsatisfied message)
    order = order + ready
    available = available | {name for s in ready for name in s.produces}
    remaining = [s for s in remaining if s not in ready]
```

Uses `+` and `|` (immutable rebinding) not `.extend()` / `.update()`.

**Rationale**: Simple and correct for the scale of this pipeline (tens of stages at most). No external dependency. Cycle detection falls out naturally — stuck means cycle or genuinely missing artifact; the error message distinguishes by reporting what each remaining stage is waiting for.

**Alternative considered**: Kahn's algorithm with explicit in-degree counting — more efficient for large graphs but adds complexity with no benefit here.

---

### 3. RNG split happens unconditionally before cache check

**Decision**:
```python
stage_rng, rng = rng.split()          # always
key = cache_key(stage, inputs, stage_rng)
outputs = load(key, cache_dir) or _run_and_save(stage, inputs, stage_rng, key, cache_dir)
```

**Rationale**: If the split happened only on a cache miss, adding a cached stage mid-graph would shift the seeds of all subsequent stages, invalidating their cache keys and breaking reproducibility. Unconditional splitting ensures the seed sequence is a pure function of the stage order, not of cache state.

---

### 4. `run` receives only filtered inputs (`stage.needs` keys)

**Decision**: `stage.run({k: artifacts[k] for k in stage.needs}, stage_rng)`

**Rationale**: Stages declare their dependencies via `needs`; receiving exactly those keys enforces the contract and prevents stages from silently depending on artifacts they didn't declare. Passing the full artifact dict would make `needs` a documentation hint rather than a real constraint.

---

### 5. Duplicate `produces` raises at validation time, not execution time

**Decision**: Before any topo-sort or execution, scan all stages for duplicate artifact names in `produces`. Raise `ValueError` immediately with both offending stage names.

**Rationale**: Fail fast at the boundary where the programmer made the mistake. Silent last-write-wins would produce subtly wrong results that are hard to debug.

---

### 6. Return all artifacts (initial + produced)

**Decision**: The returned dict contains both initial artifacts and all stage outputs.

**Rationale**: Callers (e.g., the CLI) often need both the corpus and the trained policy. Returning a subset would force callers to merge manually.

## Risks / Trade-offs

- **O(n²) topo sort** — acceptable for pipeline sizes in this project (< 20 stages). If ever needed, swap to Kahn's without changing the public API.
- **`torch.load(weights_only=False)` in persistence** — already present in W5; runner inherits this trust boundary. Not introduced here.
- **No stage-level error wrapping** — a failing `stage.run` propagates its exception directly. This is intentional: don't hide errors.

## Open Questions

None. All design decisions were resolved during exploration (see conversation context).
