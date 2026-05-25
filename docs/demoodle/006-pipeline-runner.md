# W6 — Topological Runner

## DAG-based pipeline execution

A **directed acyclic graph** (DAG) where nodes are stages and edges are artifact
dependencies is the standard model for ML pipelines. You describe *what* each
stage needs and produces; the system determines *when* to run it.

The alternative — hardcoding a linear execution order — breaks as soon as you want
to add a stage that only some pipelines need, run stages in parallel when
dependencies allow, or cache individual stages independently.

Every major ML orchestration system uses the DAG model:
- **Apache Airflow**: tasks declare `upstream_task_ids`; the scheduler resolves
  execution order and handles retries and parallelism
- **Prefect**: tasks are Python functions; dependencies are implicit in how return
  values flow between them; the runtime builds the graph
- **Metaflow**: steps declare `self.next()`; the framework manages traversal and
  can distribute steps to cloud resources
- **Luigi**: tasks implement `requires()` returning upstream tasks and `output()`
  returning target files; the scheduler runs in-process
- **Kubeflow Pipelines**: components are connected in Python; the compiler produces
  an Argo Workflow that runs on Kubernetes

Demoodle's runner is ~50 lines because it makes one simplifying assumption: stages
run sequentially in a single process. No scheduler, no distributed execution, no
retries. This is the right trade-off for a local educational tool.

## Topological sort

`_topo_sort` implements an iterative Kahn's-algorithm-style sort:

1. Start with the set of artifacts already available (`initial_artifacts`)
2. Find all stages whose `needs` are fully satisfied by the available set
3. Add them to the execution order; add their `produces` to the available set
4. Repeat until no stages remain — or raise if no progress is possible

```
initial: {corpus}

pass 1 ready: [train_tokenizer]   (needs: corpus ✓)
available:    {corpus, tokenizer}

pass 2 ready: [build_dataset]     (needs: corpus ✓, tokenizer ✓)
available:    {corpus, tokenizer, dataset}

pass 3 ready: [pretrain]          (needs: dataset ✓)
available:    {corpus, tokenizer, dataset, base_policy, metrics}
```

If a pass finds no ready stages but stages remain, the graph has a cycle or an
unsatisfied dependency — an error is raised with the specific unresolved `needs`
so the problem is easy to diagnose.

Stages at the same level are sorted alphabetically for determinism. Without this,
two stages with identical dependencies could run in any order, producing different
cache key sequences and non-deterministic test output.

## Adding a stage requires no runner edit

The runner is a pure consumer of `Stage.needs` and `Stage.produces`. It never
inspects stage names or imports any stage-specific module. A new stage is
registered simply by including it in the list passed to `runner.run()`.

This is the core plug-in payoff: W31 (the capstone test) proves it by registering
a brand-new architecture and stage and running them end-to-end with zero changes
to `runner.py`.

## RNG threading

The runner threads `RNG` through stages via `.split()`:

```python
for stage in sorted_stages:
    stage_rng, rng = rng.split()
    outputs = stage.run(inputs, stage_rng)
```

Each stage gets a child RNG derived from the current root. After splitting, `rng`
advances so the next stage gets a different child. Each stage's random draws are
a pure function of: the initial seed + the stage's position in topological order.

Reordering two stages that depend on the same upstream artifact will change their
RNG children — this is expected and desirable. Adding a new stage that comes
*before* an existing one will change that existing stage's RNG child, invalidating
its cache entry and causing a fresh run. This is also expected: the inputs to the
stage (including its RNG) have changed.
