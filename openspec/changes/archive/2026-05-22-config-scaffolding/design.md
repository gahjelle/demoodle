## Context

The project follows a Functional Core / Imperative Shell architecture. Configuration sits at the shell layer — it is loaded once at startup, validated, and then threaded as typed values into the pure core. The project already depends on `configaroo` for configuration loading; pydantic is being added to provide schema validation and static types.

W2 originally planned a core `Config` frozen dataclass. That is superseded here: per-section pydantic models serve the same purpose with less translation overhead and better runtime validation.

## Goals / Non-Goals

**Goals:**
- Single importable `config` singleton, typed as `DemoodleConfig`
- Per-axis, per-implementation config sections (e.g. `architecture.mlp`) so each component receives only the fields it understands
- One env var override (`DEMOODLE_ARCHITECTURE`) as a proof-of-concept and demo affordance
- Default TOML file committed to the repo so the project works out of the box

**Non-Goals:**
- CLI argument parsing (cyclopts handles that separately)
- Live config reloading
- Full env var coverage for every nested field (extend later as needed)

## Decisions

### 1. Nested per-implementation sections over a flat config

**Decision:** `[architecture.mlp]`, `[architecture.transformer]`, etc. — each architecture gets its own typed sub-model.

**Rationale:** `init_state(cfg: MLPConfig, rng)` is more precise than `init_state(cfg: DemoodleConfig, rng)`. Each component only sees its own fields; adding a new architecture never touches other models. The shell reads `config.architecture.active` to decide which sub-model to pass.

**Alternative considered:** Single flat `Config` with all hyperparameters. Rejected because unrelated fields leak into every architecture's signature and type checks become meaningless.

### 2. `DemoodleConfig` (pydantic) replaces the planned core `Config` frozen dataclass

**Decision:** Drop W2's `Config` frozen dataclass. Pass pydantic section models directly.

**Rationale:** Pydantic models with `model_config = ConfigDict(frozen=True)` are immutable value types. Adding a second translation layer (pydantic → frozen dataclass) would buy nothing and require keeping two schemas in sync.

### 3. `vocab_size` stays out of config

**Decision:** `vocab_size` is not a config field; it is read from the `Tokenizer` artifact at `init_state` time.

**Rationale:** `vocab_size` is derived from training data, not a user setting. Putting it in config would require a mutable update after tokenizer training, which breaks immutability.

### 4. Single env var override via explicit `add_envs` mapping

**Decision:** `add_envs({"ARCHITECTURE": "architecture.active"}, prefix="DEMOODLE_")` — explicit mapping, not automatic discovery.

**Rationale:** Automatic env discovery only covers top-level keys. Explicit mapping gives full control over which nested fields are env-overridable. Start with one (`DEMOODLE_ARCHITECTURE`); add `DEMOODLE_TOKENIZER` etc. when the tokenizer axis is built.

## Risks / Trade-offs

- **Module-level singleton** — `config` is evaluated at import time. This makes it straightforward to use but harder to swap in tests. Mitigation: tests that need a different config can patch `demoodle.config.config` or instantiate `DemoodleConfig` directly.
- **Stub sections for future architectures** — `[architecture.transformer]` fields will be placeholders until W14. Risk of stale defaults. Mitigation: mark stub sections with comments in the TOML; validate required fields when the architecture is actually used, not at load time.
- **pydantic version** — pydantic v2 has a significantly different API from v1. Pin to v2; use `model_config = ConfigDict(frozen=True)` not the v1 `class Config` inner class.

### 5. Shared `StrictModel` base class

**Decision:** Define `class StrictModel(BaseModel): model_config = ConfigDict(frozen=True, extras="forbid")` in `schemas.py` and have all config models inherit from it instead of `BaseModel` directly.

**Rationale:** `frozen=True` enforces immutability uniformly. `extras="forbid"` makes unknown TOML keys (e.g. typos) a hard error at load time rather than silent data loss. Centralising these settings in one base class means they can't be accidentally omitted from a new section model added later.

## Open Questions

None. All decisions have been resolved in the explore session.
