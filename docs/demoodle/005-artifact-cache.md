# W5 — Artifact Cache

## Content-addressed storage

A **content-addressed store** identifies objects by a hash of their content. The
key is derived from *what* something is, not *where* or *when* it was stored. The
same input always maps to the same key.

This is the foundational idea behind several build and storage tools:
- **Git**: blobs, trees, and commits are stored by SHA-1/SHA-256 of their content
- **Nix / Bazel**: build outputs are keyed by a hash of all inputs (source files +
  compiler version + flags) — the same inputs always yield the same output
- **ccache**: C/C++ compilation results are cached by a hash of source + preprocessed
  output + compiler version

ML experiment tracking tools take a softer approach. **DVC** stores datasets and
models in a content-addressed remote keyed by file hashes, with a Git-tracked
pointer file. **MLflow** identifies runs by a combination of experiment ID, run ID,
and timestamp — content-addressed for artifacts but not for runs. **Weights &
Biases** stores artifacts in a content-addressed bucket with human-readable aliases
layered on top.

## Cache key composition

Demoodle's `cache_key` function hashes six components into a single SHA-256 hex
digest:

```
stage name
+ demoodle package version
+ git commit hash (short)
+ stage config hash
+ for each input: artifact name + hash of artifact content
+ rng seed
```

Each component protects against a different class of stale cache:
- **Stage name**: prevents collision between stages that happen to have the same
  inputs (e.g. two stages that both consume only `corpus`)
- **Package version + git commit**: invalidates when code changes — a bug fix
  that changes training behavior should produce a fresh artifact
- **Config hash**: invalidates when hyperparameters change (learning rate, number
  of steps, batch size)
- **Input artifact hashes**: propagates invalidation — if an upstream stage
  produces a different artifact, everything downstream is invalidated
- **RNG seed**: invalidates when the random initialization changes

This composition mirrors Bazel's action key construction, and is similar to how
DVC builds its run cache keys.

## Hashing inputs, not outputs

The store caches *outputs* but the key is derived entirely from *inputs*. This
works because training is deterministic: the same stage, code, config, inputs, and
seed always produce the same artifacts. If you run the same stage twice with the
same key, you get the same outputs.

This assumption breaks if:
- The stage calls an external API that returns different results each time
- A library produces non-deterministic results (some GPU operations do)
- The code changes but the git commit hash hasn't been updated (uncommitted changes)

## The dirty worktree warning

The cache key includes the git commit hash. If the working tree has uncommitted
changes, the committed hash doesn't reflect the actual code — the cache might
serve a result that was produced by different code than what's currently on disk.

```python
WORKTREE_DIRTY: bool = is_worktree_dirty()
```

When `git status --porcelain` reports changes, the runner emits a warning at
startup. This is a deliberate design decision: warn rather than block. Nix takes
the stricter stance — it refuses to build with uncommitted changes (hermetic
builds). Demoodle warns rather than blocks because being blocked mid-experiment
is frustrating for development iteration.

## Artifact hashing

`_hash_artifact` converts each variant to bytes and feeds them into SHA-256:
- `Corpus`: hash the raw text bytes
- `CharTokenizer`: hash the sorted `char_to_id` mapping (sorted for stability)
- `Dataset`: reinterpret the token tensor as raw bytes via `.view(torch.uint8)`
- `Policy`: hash each weight name + raw bytes, sorted by name for stability
- `TrainingMetrics`: hash the repr of the loss list

The `.view(torch.uint8)` trick reinterprets float32 weights as their raw
4-bytes-per-element representation without copying data. It's exact — two tensors
with identical values produce identical bytes — and fast.
