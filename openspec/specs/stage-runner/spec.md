# Spec: Stage Runner

## Purpose

Orchestrates execution of a pipeline by resolving stage dependencies via topological sort, integrating with the artifact cache to skip redundant computation, and threading a deterministic RNG through every stage so that downstream seeds are stable regardless of cache state.

## Requirements

### Requirement: Execute stages in topological order
The runner SHALL accept a list of `Stage`s, a dict of initial `Artifact`s, an `RNG`, and a `Path` cache directory. It SHALL execute stages in an order consistent with each stage's `needs` and `produces` declarations and return a dict containing all initial and produced artifacts.

#### Scenario: Three stages run in dependency order
- **WHEN** stages A→B→C are passed in reverse order with A producing what B needs and B producing what C needs
- **THEN** they execute in order A, B, C regardless of input order

#### Scenario: Independent stages sort by name within a tier
- **WHEN** two stages have no dependency between them and are both immediately runnable
- **THEN** they execute in ascending lexicographic order by stage name

### Requirement: Validate no duplicate produces at call time
Before executing any stage, the runner SHALL raise `ValueError` if two or more stages declare the same artifact name in `produces`.

#### Scenario: Duplicate produces raises immediately
- **WHEN** two stages both declare `"tokenizer"` in their `produces`
- **THEN** `ValueError` is raised before any stage runs, naming both offending stages

### Requirement: Detect cycles and unsatisfiable dependencies
If the graph cannot be fully ordered (cycle or genuinely missing artifact not present in initial artifacts), the runner SHALL raise `ValueError` listing the names of all unsatisfied stages and the specific artifact names they are still waiting for.

#### Scenario: Cycle raises with unsatisfied stage info
- **WHEN** stage A needs `"x"` produced by B, and B needs `"y"` produced by A
- **THEN** `ValueError` is raised and the message includes both stage names and their missing needs

#### Scenario: Missing initial artifact raises
- **WHEN** a stage needs an artifact not in initial artifacts and not produced by any other stage
- **THEN** `ValueError` is raised naming the stage and the missing artifact

### Requirement: Thread RNG unconditionally through every stage
The runner SHALL call `rng.split()` for every stage in execution order, before checking the cache, and pass the resulting child RNG to both `cache_key` and `stage.run`.

#### Scenario: Cache state does not affect downstream seeds
- **WHEN** a stage is satisfied from cache on one run and executed fresh on another
- **THEN** all stages after it receive the same `RNG` seed in both runs

### Requirement: Filter inputs to declared needs
The runner SHALL pass to `stage.run` only the artifact keys listed in `stage.needs`, not the full available artifact dict.

#### Scenario: Stage receives only declared inputs
- **WHEN** three artifacts are available but a stage declares needs for only one
- **THEN** `stage.run` is called with a dict containing exactly that one artifact

### Requirement: Integrate with the artifact cache
The runner SHALL check the persistence cache before executing each stage. On a hit it SHALL use the cached outputs; on a miss it SHALL execute the stage and save the outputs to cache.

#### Scenario: Second run with unchanged inputs hits cache
- **WHEN** the runner is called twice with the same stages, artifacts, and RNG
- **THEN** no stage's `run` function is called on the second invocation

#### Scenario: Changed seed bypasses cache
- **WHEN** the runner is called a second time with a different root RNG seed
- **THEN** stages are re-executed and new results are saved to cache
