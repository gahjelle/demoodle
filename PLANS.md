# LLM Demo Tool — Work Items (Build Order)

> Ordered backlog. Each item is self-contained enough to drop into a prompt.
> Each lists **goal · build · done-when · depends-on**. Implement top to bottom;
> within a milestone, order matters less. Milestones M0–M2 are the day-one slice.

See VISION.md for information about the final vision, the end-goal.

These plans are not fully authoritative, question details during implementation.
Update future work items based on decisions made.

---

## Milestone 0 — Foundations (pure core types & seams)

### ✅ W1. Project skeleton & tooling
- **Goal:** empty but importable package with the directory layout from the
  Day-One doc, plus test/lint/format config.
- **Build:** `demoodle/` packages with `__init__.py`, `pyproject.toml`, PyTorch
  (CPU) + pytest pinned, a passing no-op test, README stub.
- **Done when:** `uv sync` works and `pytest` runs (even with 0 tests).
- **Depends on:** —

### ✅ W2. Core value types
- **Goal:** the immutable data spine.
- **Build:** in `core/types.py`: `Seq` alias, `Output` (logits + optional sampled
  ids), and the `Artifact` tagged union with variants `Corpus`, `Tokenizer`,
  `Dataset`, `Policy`, `TrainingMetrics` (leave `RewardModel`, `PreferenceData` as
  stubs/comments for later). **`Policy` must include an optional, nullable
  `value_head` field (`None` by default)** — reserved for PPO so we never retrofit
  architectures later.
- **Note:** `Config` is **not** a frozen dataclass here. Configuration is handled
  by `demoodle.config` (pydantic + configaroo). Import with
  `from demoodle.config import config`, or pass section models directly (e.g.
  `MLPConfig`) into `init_state()`. See `src/demoodle/config/` and
  `openspec/changes/config-scaffolding/` for details.
- **Done when:** types import; a trivial test constructs each and confirms they're
  frozen/immutable.
- **Depends on:** W1

### ✅ W3. RNG value type
- **Goal:** deterministic, explicit randomness.
- **Build:** small `RNG` wrapping a seed/key with `.split() -> (RNG, RNG)` and
  helpers to draw what training needs (e.g. a torch generator from the key). No
  global seeding.
- **Done when:** `split` is reproducible; same seed → same draws; a property test
  confirms two splits diverge but each is stable.
- **Depends on:** W2

### ✅ W4. Protocols / ports
- **Goal:** define every swap point now, even if unused.
- **Build:** in `ports/protocols.py`: `TokenizerProtocol`, `ArchitectureProtocol`,
  `InspectableProtocol` (`call` required, `explain` optional returning `{}` by
  default), and the `Stage` frozen dataclass (`name: str`, `needs: list[str]`,
  `produces: list[str]`, `run`). All protocols use the `Protocol` suffix to avoid
  shadowing artifact types in `core/types`. No `FrontendProtocol` — the real frontend
  seam is the W28 shell API.
  `Stage.run` signature includes `RNG` so stages are pure functions of their inputs:
  `Callable[[dict[str, Artifact], RNG], dict[str, Artifact]]`.
- **Done when:** protocols import; a dummy class type-checks against each.
- **Depends on:** W2, W3

---

## Milestone 1 — The Runner (imperative shell)

### ✅ W5. Artifact cache / persistence
- **Goal:** don't retrain what hasn't changed.
- **Build:** `shell/persistence.py` — content-addressed store keyed by hash of
  (stage name + code version + input artifact hashes + config + seed). Save/load
  artifacts to a cache dir.
- **Done when:** saving then loading an artifact round-trips; changing the seed
  changes the key.
- **Depends on:** W2, W3

### ✅ W6. Topo-sort runner
- **Goal:** execute a stage graph in dependency order with caching.
- **Build:** `shell/runner.py` — take a list of `Stage`s + initial artifacts
  (e.g. the corpus), topologically sort by `needs`/`produces`, execute each
  (hitting cache when possible), thread `RNG` via `.split()`, return the final
  artifact dict.
- **Done when:** unit test with three fake stages runs them in correct order;
  cycle detection raises; second run with no changes hits cache for all stages;
  **adding a stage requires no runner edit.**
- **Depends on:** W4, W5

---

## Milestone 2 — First Real Model (day-one slice complete)

### ✅ W7. Names corpus + Corpus artifact
- **Goal:** committed real data.
- **Build:** add `data/names.txt` (public-domain names, one per line); a loader
  producing a `Corpus` artifact.
- **Done when:** loader returns a `Corpus`; a test asserts non-empty and expected
  line count ballpark.
- **Depends on:** W2

### ✅ W8. CharTokenizer + train_tokenizer stage
- **Goal:** the tokenizer seam, trivial implementation.
- **Build:** `CharTokenizer` frozen dataclass (`{char:id}` mapping built from
  corpus; `encode`/`decode`/`vocab_size` satisfying `TokenizerProtocol`
  structurally — no inheritance). Add `CharTokenizer` to the `Artifact` union
  in `core/types.py`. A `train_tokenizer` stage `corpus -> tokenizer` produces
  a `CharTokenizer`. Add a `CharTokenizer` branch to `_hash_artifact` in
  `shell/persistence.py`.
- **Done when:** `decode(encode(s)) == s` for sample strings; stage produces a
  `CharTokenizer` artifact via the runner.
- **Depends on:** W4, W6, W7

### W9. build_dataset stage
- **Goal:** turn text into training pairs.
- **Build:** stage `(corpus, tokenizer) -> dataset`; encode corpus, form
  (input_char, next_char) pairs (with start/end handling), batchable.
- **Done when:** produces a `Dataset`; a test checks shapes and that targets are
  inputs shifted by one.
- **Depends on:** W8

### W10. Learned-bigram architecture
- **Goal:** the first real, trainable model.
- **Build:** `architectures/bigram.py` — V×V weight matrix. `BigramArchitecture`
  takes `vocab_size` at construction and satisfies both `ArchitectureProtocol`
  and `InspectableProtocol`. `init_state(rng: RNG) -> Policy` (config bound at
  construction, pure function of rng); `forward(policy, tokens) -> Output`
  (logits); `call(seq, temperature, top_k=None, top_p=None) -> Output` (softmax +
  sampling, returns logits and sampled token id). **Put `top_k`/`top_p` in the
  `call` seam now** even though they barely matter for the bigram — they pay off
  for the transformer + code corpus (low-temp/top-k yields more valid code) and we
  never want to retrofit the signature. `explain(seq) -> {}` (inherited default).
- **Done when:** `forward` is deterministic under fixed seed; `call` returns an
  `Output` with `sampled_ids` set; temperature changes the distribution sharpness
  (tested); top-k/top-p restrict the sampled set (tested).
- **Depends on:** W3, W4, W8

### W11. pretrain stage
- **Goal:** train the bigram for real and record it.
- **Build:** stage `(dataset, <arch/config>) -> (base_policy, metrics)`; a real
  gradient-descent loop; capture a real loss curve into `TrainingMetrics`.
- **Done when:** loss **decreases** over steps on names; produces a `Policy`;
  re-run hits cache.
- **Depends on:** W9, W10

### W12. CLI front end (train + call)
- **Goal:** a human can drive the slice.
- **Build:** `frontends/cli.py` — `train` builds & runs the day-one graph and
  prints the loss curve as text; `call --temperature` loads `base_policy` and
  prints a sampled continuation.
- **Done when:** matches the Day-One "Definition of Done": trains, caches on
  rerun, generates plausible names, temperature visibly matters.
- **Depends on:** W11

> ✅ **Day-one slice ends here.** Everything below is additive.

---

## Milestone 3 — Architecture Axis

### W13. MLP architecture
- **Goal:** add context + embeddings.
- **Build:** Bengio-style MLP (embedding table, context window, hidden layer,
  output) implementing the same `Architecture`/`Inspectable` protocol.
- **Done when:** trains via the **existing** `pretrain` stage with only a config
  change; lower loss / better samples than bigram; **no runner or stage edits.**
- **Depends on:** W10, W11

### W14. TinyShakespeare corpus
- **Goal:** a second corpus for the bigger models (moved ahead of the transformer,
  which depends on it).
- **Build:** commit `data/tinyshakespeare.txt`; a loader → `Corpus`. Selectable
  via config.
- **Done when:** runner trains any existing architecture on either corpus by
  config alone.
- **Depends on:** W7

### W15. Code corpus
- **Goal:** a third corpus specialized on code — the tokenization-axis showpiece.
- **Build:** commit `data/python_corpus.txt` and a loader → `Corpus`, selectable
  via config. Use **permissively-licensed / synthetic / self-written Python**
  (PSF-licensed stdlib excerpts, or a generated templated corpus) — **not scraped
  repos.** **Reuses the existing byte-level BPE (W18) — no new tokenizer is added**
  (it is byte-level, so it ingests arbitrary code unchanged; an external tokenizer
  like tiktoken would break the "we built it in ~80 readable lines" premise and
  kill the W20 merge-inspection payoff). Optionally support a self-referential
  "train Demoodle on its own source" corpus as a selectable Easter egg.
- **Done when:** runner trains any architecture on the code corpus by config alone;
  BPE merges learned on code differ visibly from names/Shakespeare (4-space indent,
  `def `, `self.`, `):` …), feeding the W20 comparison.
- **Depends on:** W7
- **Note:** measure the source first
  (`find . -name '*.py' | xargs wc -c`). TinyShakespeare is ~1 MB; a single codebase
  is likely ~100–300 KB — trainable but prone to memorization, which dulls the
  "learning structure" lesson and makes the W21 validity eval meaningless. Augment
  with stdlib/synthetic Python toward ~1 MB if you want generalization. A toy CPU
  transformer won't emit runnable code — expect code-shaped text with plausible
  indentation/keywords (a fine demo if framed honestly). If "it actually parses"
  matters more than the tokenization story, a tiny regular DSL beats Python — but
  that's a deliberate trade-off, not the default.

### W16. Tiny transformer architecture
- **Goal:** the real mechanism.
- **Build:** 1–2 layer, 1–2 head transformer (token + positional embeddings,
  self-attention, residual stream, MLP block) under the same protocol; small
  enough for CPU/MPS in minutes.
- **Done when:** trains via existing `pretrain` on names and TinyShakespeare (and
  optionally the code corpus); produces recognizably structured output.
- **Depends on:** W13, W14

### W17. Attention inspection (`explain`)
- **Goal:** make attention visible.
- **Build:** implement `explain()` on the transformer to return attention weights
  / per-position info; bigram & MLP still return `{}`.
- **Done when:** `explain` yields attention matrices of correct shape; front ends
  can feature-detect presence. CLI: expose an `attn` verb that renders the matrix as
  a text heatmap.
- **Depends on:** W16

---

## Milestone 4 — Tokenization Axis

### W18. Byte-level BPE tokenizer
- **Goal:** real, learned tokenization.
- **Build:** `BpeTokenizer` frozen dataclass — ~50–80-line byte-level BPE
  satisfying `TokenizerProtocol` structurally; learns merges from a corpus.
  Add `BpeTokenizer` to the `Artifact` union in `core/types.py` and add a
  `BpeTokenizer` branch to `_hash_artifact` in `shell/persistence.py`.
  No pre-tokenizer — merging across "real" code boundaries (e.g. `):` or
  `):\n    `) is fine and illustrative.
- **Done when:** round-trips text; trains on TinyShakespeare in seconds; vocab
  size configurable.
- **Depends on:** W8

### W19. BPE in train_tokenizer stage
- **Goal:** swap tokenizers with no downstream changes.
- **Build:** make `train_tokenizer` produce either Char or BPE by config; verify
  embedding table resizes to `vocab_size` and sequences shorten.
- **Done when:** transformer trains on BPE tokens via the **existing** graph;
  **no model/runner edits** beyond config.
- **Depends on:** W18, W16

### W20. Tokenization inspection
- **Goal:** the teaching payoff.
- **Build:** expose, via the tokenizer's `explain`/helpers: the merge timeline,
  same-string segmentation under char vs BPE with token counts, and the
  "count the letters" failure illustration. **Use a code snippet as the headline
  example** — indentation and keyword merges, and the char-vs-BPE token-count gap
  (largest on code). Add a per-corpus "top merges" view so names / Shakespeare /
  code merges can be compared side by side.
- **Done when:** a front end can render merges step-by-step and a char-vs-BPE
  comparison for an arbitrary string, including a code example. CLI: expose `tokens`
  (segmentation + counts under char vs bpe) and `merges` (ranked/animated merge list,
  comparable across corpora) verbs.
- **Depends on:** W19

---

## Milestone 5 — Training / Post-training Axis

### W21. Generation-quality eval
- **Goal:** a concrete, non-loss quality signal so post-training improvement is
  measurable (loss-down alone is a weak demo).
- **Build:** a small eval helper/stage that samples N generations from a policy and
  scores **corpus-appropriate validity**, recorded into `EvalMetrics`:
  for **code**, the fraction that `compile()` / `ast.parse()` without error;
  for **names**, the fraction matching a plausible-name heuristic;
  for **Shakespeare**, line-length / structure stats. Pluggable per corpus.
- **Done when:** produces a scalar (e.g. syntax-validity rate) deterministically
  under a seed; a baseline is measurable before post-training; consumed by the
  W22 / W26 done-whens to show a *rise*. CLI: expose an `eval` verb, including an
  `--across <stages>` mode that prints a cross-stage progression table (e.g.
  `pretrain 34% → sft 51% → dpo 68%`).
- **Depends on:** W11, W15 (works on any corpus; code is the headline)

### W22. SFT stage
- **Goal:** shape behavior with curated pairs.
- **Build:** a small (prompt → desired output) dataset for a toy task; `sft` stage
  `base_policy -> sft_policy` fine-tuning on it.
- **Done when:** `sft_policy` measurably shifts toward the desired format vs base
  (and the W21 validity metric rises where applicable); runs through the existing
  runner.
- **Depends on:** W11 (and ideally W16)

### W23. Toy reward + auto-generated PreferenceData
- **Goal:** a real, computable preference signal.
- **Build:** a reward function (e.g. "prefer vowel-initial / shorter / period-
  ending"); generate `PreferenceData` by sampling a policy and labeling pairs with
  it. Promote `RewardModel` & `PreferenceData` from stubs to real members of the
  `Artifact` union (from W2).
- **Done when:** preference pairs generate deterministically under a seed; reward
  function unit-tested.
- **Depends on:** W2, W11

### W24. Interactive human-labeling (live preference collection)
- **Goal:** put a real human in the loop for a handful of pairs — the RLHF "aha".
- **Build:** a **shell/front-end** flow (CLI prompt first; richer in web later)
  that samples two completions, asks "which is better?", collects ~20–40 clicks,
  and emits a `PreferenceData` artifact. Support a **blend mode**: seed with W23
  auto-pairs, then let the human add/override a few. Human interaction stays in the
  shell; only the resulting data crosses into core.
- **Done when:** a session produces a valid `PreferenceData` artifact consumable by
  W25 unchanged; blend mode merges human + auto pairs; the demo surfaces the honest
  caveat ("~30 labels is noisy; real systems use tens of thousands"). CLI: expose a
  `label` verb (prints A/B, reads 1/2 from stdin; `--blend` to seed from auto-pairs).
- **Depends on:** W23

### W25. Reward model stage
- **Goal:** learn the scalar scorer (the human feedback becoming weights).
- **Build:** `train_reward_model` stage `preference_data -> reward_model`; a small
  head/scorer trained on the pairs (works on auto, human, or blended data).
- **Done when:** RM scores preferred > rejected on held-out pairs above chance;
  produces a `RewardModel` artifact; retrains in seconds on ~30 human labels.
- **Depends on:** W23 (W24 optional but recommended)

### W26. DPO stage (default RLHF)
- **Goal:** the headline post-training step, simplest stable form.
- **Build:** `rlhf` stage with `needs=("sft_policy", "preference_data")`, reading
  `sft_policy` as a **frozen reference**; implement the DPO loss directly on
  preference pairs (no reward model consumed). `value_head` stays `None`.
- **Done when:** policy behavior shifts toward the preferred direction (and the
  W21 metric reflects it); KL-to-reference tracked; visible in `call`; produces
  `rlhf_policy`. Confirms the artifact graph (not a linear fold) was right.
- **Depends on:** W22, W23

### W27. PPO stage (classical RLHF variant)
- **Goal:** the full pipeline + reward/KL dashboard.
- **Build:** `rlhf_ppo` stage with `needs=("sft_policy", "reward_model")`, frozen
  reference for KL control; activate the `Policy` **value head**; sample → score
  with RM → advantage → clipped policy-gradient update. A drop-in alternative
  stage, not a rewrite.
- **Done when:** mean reward **rises** while KL is tracked; behavior shift in
  `call`; produces `rlhf_ppo_policy`; demonstrates the value-head slot paying off.
- **Depends on:** W25, W26

---

## Milestone 6 — Inspection & Showpiece Front Ends

### W28. Metrics & generation inspection API
- **Goal:** one clean shell API for all visual data.
- **Build:** shell functions returning loss curves, per-step token distributions
  during generation, attention (via `explain`), RM scores, RLHF reward/KL traces,
  and W21 generation-quality scores — as plain data. For the live front ends, also
  provide: (a) a **streaming generation** call that *yields* per-token distributions
  as it samples (not just the final string), so the TUI/web can draw probability
  bars token-by-token; (b) a **`compare(config_a, config_b)`** helper returning
  aligned before/after data (samples, loss, eval, etc.) for split-screen views;
  (c) **per-node cache hit/miss** surfaced from the runner so a front end can color
  the stage graph (small W6 addition — return a hit/miss map alongside the artifact
  dict; does not change the cache key).
- **Done when:** CLI can dump any of these as text/JSON; streaming yields
  incrementally; `compare` returns aligned A/B data; cache hit/miss is queryable;
  no front end touches core. CLI: expose `compare` (renders aligned A/B as a
  diff/side-by-side block) and `graph` (prints the DAG with per-node cache-hit
  markers) verbs.
- **Depends on:** W6, W12, W17, W21, W26, W27

### W29. TUI front end
- **Goal:** a rich terminal interface for the demo — a live instrument, not a
  read-only dashboard. See DEMO_TUI.md for the full layout, act-by-act script, and
  restart model.
- **Build:** a TUI (e.g. Textual) that calls **only** the W28 API and provides:
  a **baseline bar** (architecture × tokenizer × corpus × training-stage) with a
  lock/unlock-one-axis state machine and "promote current to baseline"; **compare
  mode** (split-screen before/after via `compare()`); a **stage-graph widget** that
  colors nodes by cache hit / running / fresh; streamed generation with per-token
  probability bars and a live temperature slider; and the context-sensitive inspect
  panel (loss curve, attention heatmap, merge timeline, RLHF reward/KL, W21 gauge).
- **Done when:** a presenter can drive all three axes from the terminal with visible
  live updates; switching the active axis re-locks the others and rehydrates from
  cache; uses only the shell/W28 API.
- **Depends on:** W28

### W30. Web front end (interactive showpiece)
- **Goal:** the live-demo UI.
- **Build:** pick architecture × tokenizer × training point; type a prompt; watch
  token-by-token probabilities with a temperature slider; attention view when
  available; RLHF reward/KL dashboard; **the live human-labeling picker (W24) as a
  rich two-completions UI**.
- **Done when:** a presenter can move along any single axis live and the effect is
  visible; a human can label preferences in-browser and watch the shift; uses only
  the shell/W28 API.
- **Depends on:** W28, W24

### W31. Plug-and-play proof (capstone test)
- **Goal:** prove the thesis mechanically.
- **Build:** a test/example that adds a brand-new trivial architecture **and** a
  trivial new stage using only the public protocols.
- **Done when:** both run end-to-end through the existing runner and a front end
  with **zero edits** to `runner.py`, existing stages, or existing architectures.
  CLI: expose a `plug <module>` verb that registers the new arch/stage so it's
  immediately runnable via the normal flags (the two-command live proof).
- **Depends on:** W6, W12 (meaningful once M3–M5 exist)

### W32. Demo harness
- **Goal:** make a live presentation bulletproof — for **both** the CLI and TUI
  — without leaking presentation polish into core or the W29 widgets. See
  docs/demo_scripts/demo_tui.md §6 and docs/demo_scripts/demo_cli.md §6.
- **Build:** a thin shell-side harness (`demoodle demo …`): a **warmup** command
  (shared by both front ends) that materializes every config the scripts visit (so all
  but the intentionally-live beats are cache hits); a **scripted-path** mode (advance /
  jump-to-act with title card + config + prompt preset per beat) for the TUI; a
  committed **`demo.sh` stepper** (the CLI arc as a commented, Enter-stepped,
  asciinema-friendly script); and a **fallback-sample** cache (a known-good rehearsed
  sample per beat) plus undo/panic actions. Pure consumer of W28.
- **Done when:** `warmup` makes a full run of both scripted paths hit cache end-to-end
  except the live beats; the TUI path is drivable one beat at a time and jumpable by
  act; `demo.sh` steps the CLI arc cleanly; a wobbly live retrain can fall back to a
  rehearsed sample without breaking flow.
- **Depends on:** W28, W29
