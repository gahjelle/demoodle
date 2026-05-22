## 1. Graph validation

- [x] 1.1 Write failing test: duplicate `produces` raises `ValueError` naming both stages
- [x] 1.2 Implement `_validate_no_duplicate_produces(stages)` — scan all stages, raise on collision
- [x] 1.3 Write failing test: cycle raises `ValueError` with unsatisfied stage names and missing needs
- [x] 1.4 Write failing test: missing initial artifact raises `ValueError` naming the stage and artifact

## 2. Topological sort

- [x] 2.1 Write failing test: three stages given in reverse order execute in correct dependency order
- [x] 2.2 Write failing test: independent stages within a tier sort by name
- [x] 2.3 Implement `_topo_sort(stages, initial_artifact_names)` using immutable ready-set extraction (`+`, `|`)
- [x] 2.4 Wire cycle/stuck detection into `_topo_sort` (satisfies 1.3 and 1.4 tests)

## 3. Core runner

- [x] 3.1 Write failing test: `run` calls `stage.run` with only the keys in `stage.needs`
- [x] 3.2 Write failing test: `run` returns dict containing both initial and produced artifacts
- [x] 3.3 Write failing test: `rng.split()` is called before cache check — downstream seeds identical regardless of cache state
- [x] 3.4 Implement `run(stages, initial_artifacts, rng, cache_dir)` — validate, topo-sort, execute loop with unconditional RNG split and cache integration

## 4. Cache integration

- [x] 4.1 Write failing test: second run with identical inputs calls no `stage.run` (all cache hits)
- [x] 4.2 Write failing test: changed root RNG seed causes re-execution
- [x] 4.3 Verify cache integration works end-to-end with the three-stage test fixture

## 5. Verification and docs

- [x] 5.1 `uv run ruff format src/ tests/`
- [x] 5.2 `uv run ruff check src/ tests/`
- [x] 5.3 `uv run ty check src/ tests/`
- [x] 5.4 `uv run pytest` — all tests pass
- [x] 5.5 Mark W6 done (✅) in `PLANS.md`
- [x] 5.6 Review `agents/README.md` and update if new files warrant an entry
