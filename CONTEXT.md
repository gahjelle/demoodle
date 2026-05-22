# Demoodle — Domain Glossary

> This file is a glossary only. No implementation details, no specs, no scratch notes.
> When a term is used in code, docs, or issues, use the definition here.

---

## Artifact

An immutable, named value that flows between **Stages** and is stored in the **Artifact Cache**. Every value a stage produces or consumes is an artifact. Artifacts are the unit of caching.

The complete set of artifact types is the `Artifact` union in `core/types.py`. It grows when new work items confirm new concrete types; types join the union because stages produce and consume them, not through inheritance.

**Current types:** `Corpus`, `CharTokenizer` (W8), `BpeTokenizer` (W18), `Dataset`, `Policy`, `TrainingMetrics`, `EvalMetrics` (W21), `RewardModel` (W23), `PreferenceData` (W23).

**Avoid:** "data", "output", "result" when referring to inter-stage values — use "artifact".

---

## Artifact Cache

The content-addressed store in `shell/persistence.py`. Keyed by a hash of: stage name + code version (package version + git short ID) + `config_hash` + sorted input artifact hashes + RNG seed. A cache hit means a stage's `run` is not called; the runner threads RNG unconditionally regardless of cache state.

---

## Stage

A frozen dataclass (`ports/protocols.py`) declaring what artifacts it **needs** and **produces** by name, a `config_hash`, and a `run` callable. Stages are pure functions of their inputs: `run(inputs: dict[str, Artifact], rng: RNG) -> dict[str, Artifact]`. The **Runner** topologically sorts and executes them.

**`needs` and `produces`** are `list[str]` (artifact key names), not the artifact types themselves.

---

## Runner

`shell/runner.py`. Accepts a list of `Stage`s in any order, topologically sorts them by `needs`/`produces`, executes each (checking the **Artifact Cache** first), threads **RNG** via `.split()` before every stage unconditionally, and returns all artifacts. Raises `ValueError` on duplicate `produces`, cycles, or unsatisfiable dependencies.

Within a dependency tier (stages with no ordering constraint between them), execution order is **alphabetical by stage name**. This is a contract, not an implementation detail — it guarantees stable, reproducible execution without requiring explicit ordering annotations.

---

## RNG

An immutable value type (`core/rng.py`) wrapping a single integer seed. Split via `.split() -> (RNG, RNG)` (JAX-style) to derive independent child RNGs. Never mutated; never stored globally. The **Runner** calls `.split()` once per stage in execution order, before any cache check, so downstream seeds are stable regardless of cache state.

---

## Protocol

A structural interface (`ports/protocols.py`) defining a behavioral contract. Classes satisfy a protocol by implementing the required methods — no inheritance needed. Protocol names carry the `Protocol` suffix to avoid shadowing artifact types with the same name (e.g., `TokenizerProtocol` vs the `CharTokenizer`/`BpeTokenizer` artifacts).

**Named protocols:**
- `TokenizerProtocol` — `encode(text) -> list[int]`, `decode(ids) -> str`, `vocab_size: int`
- `ArchitectureProtocol` — `init_state(rng: RNG) -> Policy` (config bound at construction), `forward(policy, tokens) -> Output`
- `InspectableProtocol` — `call(seq, temperature, top_k=None, top_p=None) -> Output`, `explain(seq) -> dict` (optional, defaults to `{}`)

---

## Policy

The artifact holding a trained model: a frozen dataclass with `model: nn.Module` and `value_head: nn.Module | None`. Write-once by convention — training always produces a new `Policy`, never mutates an existing one. The `value_head` slot is reserved for PPO (W27) and is `None` for all other training regimes.

---

## InspectableProtocol

The inference and inspection interface. `call(seq, temperature, top_k=None, top_p=None) -> Output` returns the full probability distribution (logits) plus the sampled next token, allowing front ends to visualise the distribution without a second forward pass. `explain(seq) -> dict` is a pure function returning arch-specific internals (e.g. attention weights for transformers); it re-runs inference internally and carries no mutable state. Architectures without meaningful internals inherit the default `explain` returning `{}` by subclassing `InspectableProtocol`.

---

## Output

Frozen dataclass: `logits: torch.Tensor` (the full next-token distribution) and `sampled_ids: torch.Tensor | None`. Returned by `ArchitectureProtocol.forward` and `InspectableProtocol.call`.

---

## Axis

One of the four independently configurable dimensions of a model configuration: **architecture**, **tokenizer**, **corpus**, and **training regime** (pretrain through post-training). The demo's core mechanic is holding three axes fixed while moving one — so the audience sees the contribution of each dimension in isolation.

**Avoid:** "knob", "dimension", "parameter" when referring to these four — use "axis" (plural: **axes**).

---

## Corpus

Artifact: raw, unsegmented text loaded from disk. The starting point of the pipeline. Also one of the four configurable **Axes** — independently selectable alongside architecture, tokenizer, and training regime. Corpus and tokenizer are correlated in practice (some tokenizers suit some corpora better) but are technically independent axes.

**Current corpora:** names (W7), TinyShakespeare (W14), code/Python (W15).

---

## Dataset

Artifact: the corpus encoded as a flat 1D tensor of token IDs. Produced by `build_dataset` from a corpus + tokenizer. Windowing into batches is the architecture's job — `Dataset` is architecture-agnostic.

---

## TrainingMetrics

Artifact: `losses: list[float]` — the per-step training loss curve from a training stage. Produced by training stages (pretrain, SFT, DPO, PPO) alongside the `Policy`.

**Avoid:** the bare name "Metrics" — use `TrainingMetrics` to distinguish from `EvalMetrics`.

---

## EvalMetrics

Artifact: the generation-quality signal produced by the eval stage (W21). Captures corpus-appropriate validity — e.g. `ast.parse` pass-rate for code, plausible-name rate for names, structure stats for Shakespeare. Produced by a separate eval stage that takes a `Policy` as input; has no dependency on `TrainingMetrics`. The validity scalar is deterministic under a fixed seed, so it is fully cacheable and comparable across training regimes.

---

## Core / Shell / Front End

The three layers of the architecture:

- **Core** (`core/`, and stage logic): pure and deterministic. Data in, data out. No disk, no clock, no printing. RNG is passed explicitly.
- **Shell** (`shell/`): owns all I/O, RNG sourcing, caching, and orchestration. The **Runner** and **Artifact Cache** live here.
- **Front ends** (`frontends/`): interchangeable views (CLI, TUI, web). All consume the shell's API; none touch core directly.
