## Context

The pipeline runner (W6) needs to skip stages whose inputs haven't changed. Stages can be expensive (training loops), so a content-addressed cache is the right primitive: the same inputs always produce the same outputs, so we can identify "already done" by hashing everything that determines the output.

W4 delivered `Stage` as a frozen dataclass. W5 extends it with a `config_hash` field and delivers `shell/persistence.py` as the cache implementation. W6 will call into persistence directly.

## Goals / Non-Goals

**Goals:**
- Content-addressed cache keyed by all inputs that determine a stage's output
- Round-trip save/load for every current `Artifact` type
- Cache miss returns `None` (caller decides what to do — W6 will run the stage)
- Changing any key component (seed, config, input artifact, code version) changes the cache key

**Non-Goals:**
- Cache eviction, size limits, or TTL — users clear the cache dir manually
- Cross-machine sharing or remote caches
- Human-readable cache format
- Concurrent cache access safety (single process, single writer assumed)

## Decisions

### Decision: `config_hash: str` on `Stage` with no default

**Chosen:** Add a required `config_hash: str` field to the `Stage` frozen dataclass. Stage authors compute it from relevant pydantic sub-configs using `model_dump_json()` and pass it at construction time. Stages with no config sensitivity pass `config_hash=""` explicitly.

**Alternatives considered:**

- *No config in the cache key (manual busting only):* Simple, but silently reuses stale cache when hyperparameters change during live exploration. Rejected because demos involve frequent hyperparameter variation.
- *Full `DemoodleConfig` in every key:* Automatic but too coarse — changing `paths.cache_dir` would invalidate all training caches. Rejected.
- *Runner threads config into `cache_key`:* Couples `persistence.py` to `DemoodleConfig` and requires stages to declare which config sections they use. More machinery for no gain over the Stage-carried hash. Rejected.

**Why no default:** Forcing `config_hash` at construction time prevents authors from silently omitting it. The cost is one extra argument at Stage construction; the benefit is explicit, auditable config sensitivity.

### Decision: `__version__` + git short ID for code version

**Chosen:** `demoodle.__version__` concatenated with the git short commit hash (via `subprocess`). If git is unavailable (not installed, not a git repo, PyPI install), fall back to `""`. The combined string is computed once at module import time.

**Why this is safe for PyPI users:** A PyPI release is immutable — `__version__` alone uniquely identifies the build. The git ID adds precision for development builds where the version string hasn't changed but the code has.

**Alternatives considered:**

- *`inspect.getsource(stage.run)`:* Doesn't capture closure values (e.g., a config value captured in a lambda). Also fragile to whitespace changes. Rejected.
- *Manual version string per stage:* Easy to forget to bump. Rejected in favour of automatic code-version detection.

### Decision: Per-type artifact hashing, not pickle hash

**Chosen:** Type-dispatch on `Artifact` to produce a stable byte representation before hashing:
- `Corpus` → `sha256(text.encode())`
- `Metrics` → `sha256(repr(losses).encode())`
- `Dataset` → `sha256(tokens.cpu().numpy().tobytes())`
- `Tokenizer` → `sha256(str(vocab_size).encode())`
- `Policy` → `sha256` over `sorted(state_dict.items())`, each `(key.encode() + param.cpu().numpy().tobytes())`

**Why not hash the pickle bytes:** Pickle output is not stable across Python versions or even across runs for some objects. Content hashing must be deterministic across processes.

### Decision: `torch.save` / `torch.load` for serialization

**Chosen:** Use `torch.save` and `torch.load` to persist artifact dicts. This handles `nn.Module`, tensors, and plain Python dataclasses in a single call with no extra dependencies.

**Trade-off:** Pickle-based, so not human-readable and sensitive to Python version changes. Acceptable for a local dev cache in a demo tool. If the cache becomes corrupt, delete and re-run.

### Decision: Public API as three pure-ish functions

```python
def cache_key(stage: Stage, inputs: dict[str, Artifact], rng: RNG) -> str: ...
def save(key: str, artifacts: dict[str, Artifact], cache_dir: Path) -> None: ...
def load(key: str, cache_dir: Path) -> dict[str, Artifact] | None: ...
```

`cache_key` is a pure function. `save` and `load` are the thin IO boundary. The runner (W6) orchestrates them; persistence.py has no runner knowledge.

## Risks / Trade-offs

- **Stale cache on dirty worktree** → The git ID reflects the last commit, not uncommitted changes. Mitigation: clear the cache dir manually when code is dirty. Accepted by both developers.
- **`Tokenizer` hash loses information** → Current `Tokenizer` only carries `vocab_size`; two different tokenizer implementations with the same vocab size hash identically. Mitigation: `CharTokenizer` (W8) will be a separate class satisfying `TokenizerProtocol`, not a `Tokenizer` subclass — so this artifact type won't hold a full tokenizer in practice. Revisit if this assumption changes.
- **Policy hashing is expensive for large models** → Iterating all parameter bytes for a large model is slow. Acceptable for the small demo models (bigram, MLP, tiny transformer). Not a concern in scope.

## Open Questions

None — all design decisions were resolved during exploration.
