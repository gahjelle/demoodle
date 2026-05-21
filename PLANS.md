# LLM Demo Tool — Work Items (Build Order)

> Ordered backlog. Each item is self-contained enough to drop into a prompt.
> Each lists **goal · build · done-when · depends-on**. Implement top to bottom;
> within a milestone, order matters less. Milestones M0–M2 are the day-one slice.

---

## Milestone 0 — Foundations (pure core types & seams)

### W1. Project skeleton & tooling
- **Goal:** empty but importable package with the directory layout from the
  Day-One doc, plus test/lint/format config.
- **Build:** `demoodle/` packages with `__init__.py`, `pyproject.toml`, PyTorch
  (CPU) + pytest pinned, a passing no-op test, README stub.
- **Done when:** `uv sync` works and `pytest` runs (even with 0 tests).
- **Depends on:** —

### W2. Core value types
- **Goal:** the immutable data spine.
- **Build:** in `core/types.py`: `Config` (frozen dataclass), `Seq` alias,
  `Output` (logits + optional sampled ids), and the `Artifact` tagged union with
  variants `Corpus`, `Tokenizer`, `Dataset`, `Policy`, `Metrics` (leave
  `RewardModel`, `PreferenceData` as stubs/comments for later). **`Policy` must
  include an optional, nullable `value_head` field (`None` by default)** — reserved
  for PPO so we never retrofit architectures later.
- **Done when:** types import; a trivial test constructs each and confirms they're
  frozen/immutable.
- **Depends on:** W1

### W3. RNG value type
- **Goal:** deterministic, explicit randomness.
- **Build:** small `RNG` wrapping a seed/key with `.split() -> (RNG, RNG)` and
  helpers to draw what training needs (e.g. a torch generator from the key). No
  global seeding.
- **Done when:** `split` is reproducible; same seed → same draws; a property test
  confirms two splits diverge but each is stable.
- **Depends on:** W2

### W4. Protocols / ports
- **Goal:** define every swap point now, even if unused.
- **Build:** in `ports/protocols.py`: `Tokenizer`, `Architecture`, `Inspectable`
  (`call` required, `explain` optional returning `{}` by default), `Frontend`,
  and the `Stage` frozen dataclass (`name`, `needs`, `produces`, `run`).
- **Done when:** protocols import; a dummy class type-checks against each.
- **Depends on:** W2

---

## Milestone 1 — The Runner (imperative shell)

### W5. Artifact cache / persistence
- **Goal:** don't retrain what hasn't changed.
- **Build:** `shell/persistence.py` — content-addressed store keyed by hash of
  (stage name + code version + input artifact hashes + config + seed). Save/load
  artifacts to a cache dir.
- **Done when:** saving then loading an artifact round-trips; changing the seed
  changes the key.
- **Depends on:** W2, W3

### W6. Topo-sort runner
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

### W7. Names corpus + Corpus artifact
- **Goal:** committed real data.
- **Build:** add `data/names.txt` (public-domain names, one per line); a loader
  producing a `Corpus` artifact.
- **Done when:** loader returns a `Corpus`; a test asserts non-empty and expected
  line count ballpark.
- **Depends on:** W2

### W8. CharTokenizer + train_tokenizer stage
- **Goal:** the tokenizer seam, trivial implementation.
- **Build:** `CharTokenizer` (build `{char:id}` from corpus; `encode`/`decode`/
  `vocab_size`) and a `train_tokenizer` stage `corpus -> tokenizer`.
- **Done when:** `decode(encode(s)) == s` for sample strings; stage produces a
  `Tokenizer` artifact via the runner.
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
- **Build:** `architectures/bigram.py` — V×V weight matrix; `init_state`,
  `forward(policy, tokens) -> Output` (logits); implement `Inspectable.call`
  (next-token softmax + temperature sampling); `explain` returns `{}`.
- **Done when:** `forward` is deterministic under fixed seed; `call` samples valid
  token ids; temperature changes the distribution sharpness (tested).
- **Depends on:** W3, W4, W8

### W11. pretrain stage
- **Goal:** train the bigram for real and record it.
- **Build:** stage `(dataset, <arch/config>) -> (base_policy, metrics)`; a real
  gradient-descent loop; capture a real loss curve into `Metrics`.
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

### W14. Tiny transformer architecture
- **Goal:** the real mechanism.
- **Build:** 1–2 layer, 1–2 head transformer (token + positional embeddings,
  self-attention, residual stream, MLP block) under the same protocol; small
  enough for CPU/MPS in minutes.
- **Done when:** trains via existing `pretrain` on names and TinyShakespeare;
  produces recognizably structured output.
- **Depends on:** W13

### W15. Attention inspection (`explain`)
- **Goal:** make attention visible.
- **Build:** implement `explain()` on the transformer to return attention weights
  / per-position info; bigram & MLP still return `{}`.
- **Done when:** `explain` yields attention matrices of correct shape; front ends
  can feature-detect presence.
- **Depends on:** W14

### W16. TinyShakespeare corpus
- **Goal:** a second corpus for the bigger models.
- **Build:** commit `data/tinyshakespeare.txt`; a loader → `Corpus`. Selectable
  via config.
- **Done when:** runner trains any architecture on either corpus by config alone.
- **Depends on:** W7

---

## Milestone 4 — Tokenization Axis

### W17. Byte-level BPE tokenizer
- **Goal:** real, learned tokenization.
- **Build:** ~50–80-line byte-level BPE implementing the `Tokenizer` protocol;
  learns merges from a corpus.
- **Done when:** round-trips text; trains on TinyShakespeare in seconds; vocab
  size configurable.
- **Depends on:** W8

### W18. BPE in train_tokenizer stage
- **Goal:** swap tokenizers with no downstream changes.
- **Build:** make `train_tokenizer` produce either Char or BPE by config; verify
  embedding table resizes to `vocab_size` and sequences shorten.
- **Done when:** transformer trains on BPE tokens via the **existing** graph;
  **no model/runner edits** beyond config.
- **Depends on:** W17, W14

### W19. Tokenization inspection
- **Goal:** the teaching payoff.
- **Build:** expose, via the tokenizer's `explain`/helpers: the merge timeline,
  same-string segmentation under char vs BPE with token counts, and the
  "count the letters" failure illustration.
- **Done when:** a front end can render merges step-by-step and a char-vs-BPE
  comparison for an arbitrary string.
- **Depends on:** W18

---

## Milestone 5 — Training / Post-training Axis

### W20. SFT stage
- **Goal:** shape behavior with curated pairs.
- **Build:** a small (prompt → desired output) dataset for a toy task; `sft` stage
  `base_policy -> sft_policy` fine-tuning on it.
- **Done when:** `sft_policy` measurably shifts toward the desired format vs base;
  runs through the existing runner.
- **Depends on:** W11 (and ideally W14)

### W21. Toy reward + auto-generated PreferenceData
- **Goal:** a real, computable preference signal.
- **Build:** a reward function (e.g. "prefer vowel-initial / shorter / period-
  ending"); generate `PreferenceData` by sampling a policy and labeling pairs with
  it. Promote `RewardModel` & `PreferenceData` from stubs to real members of the
  `Artifact` union (from W2).
- **Done when:** preference pairs generate deterministically under a seed; reward
  function unit-tested.
- **Depends on:** W2, W11

### W22. Interactive human-labeling (live preference collection)
- **Goal:** put a real human in the loop for a handful of pairs — the RLHF "aha".
- **Build:** a **shell/front-end** flow (CLI prompt first; richer in web later)
  that samples two completions, asks "which is better?", collects ~20–40 clicks,
  and emits a `PreferenceData` artifact. Support a **blend mode**: seed with W21
  auto-pairs, then let the human add/override a few. Human interaction stays in the
  shell; only the resulting data crosses into core.
- **Done when:** a session produces a valid `PreferenceData` artifact consumable by
  W23 unchanged; blend mode merges human + auto pairs; the demo surfaces the honest
  caveat ("~30 labels is noisy; real systems use tens of thousands").
- **Depends on:** W21

### W23. Reward model stage
- **Goal:** learn the scalar scorer (the human feedback becoming weights).
- **Build:** `train_reward_model` stage `preference_data -> reward_model`; a small
  head/scorer trained on the pairs (works on auto, human, or blended data).
- **Done when:** RM scores preferred > rejected on held-out pairs above chance;
  produces a `RewardModel` artifact; retrains in seconds on ~30 human labels.
- **Depends on:** W21 (W22 optional but recommended)

### W24. DPO stage (default RLHF)
- **Goal:** the headline post-training step, simplest stable form.
- **Build:** `rlhf` stage with `needs=("sft_policy", "preference_data")`, reading
  `sft_policy` as a **frozen reference**; implement the DPO loss directly on
  preference pairs (no reward model consumed). `value_head` stays `None`.
- **Done when:** policy behavior shifts toward the preferred direction; KL-to-
  reference tracked; visible in `call`; produces `rlhf_policy`. Confirms the
  artifact graph (not a linear fold) was right.
- **Depends on:** W20, W21

### W25. PPO stage (classical RLHF variant)
- **Goal:** the full pipeline + reward/KL dashboard.
- **Build:** `rlhf_ppo` stage with `needs=("sft_policy", "reward_model")`, frozen
  reference for KL control; activate the `Policy` **value head**; sample → score
  with RM → advantage → clipped policy-gradient update. A drop-in alternative
  stage, not a rewrite.
- **Done when:** mean reward **rises** while KL is tracked; behavior shift in
  `call`; produces `rlhf_ppo_policy`; demonstrates the value-head slot paying off.
- **Depends on:** W23, W24

---

## Milestone 6 — Inspection & Showpiece Front Ends

### W26. Metrics & generation inspection API
- **Goal:** one clean shell API for all visual data.
- **Build:** shell functions returning loss curves, per-step token distributions
  during generation, attention (via `explain`), RM scores, and RLHF reward/KL
  traces — as plain data.
- **Done when:** CLI can dump any of these as text/JSON; no front end touches core.
- **Depends on:** W12, W15, W24, W25

### W27. Notebook front end
- **Goal:** inline plots.
- **Build:** a notebook implementing `Frontend` that calls the W26 API and plots
  loss curves, attention heatmaps, BPE merges, RLHF dashboards.
- **Done when:** one notebook walks all three axes with real charts.
- **Depends on:** W26

### W28. Web front end (interactive showpiece)
- **Goal:** the live-demo UI.
- **Build:** pick architecture × tokenizer × training point; type a prompt; watch
  token-by-token probabilities with a temperature slider; attention view when
  available; RLHF reward/KL dashboard; **the live human-labeling picker (W22) as a
  rich two-completions UI**.
- **Done when:** a presenter can move along any single axis live and the effect is
  visible; a human can label preferences in-browser and watch the shift; uses only
  the shell/W26 API.
- **Depends on:** W26, W22

### W29. Plug-and-play proof (capstone test)
- **Goal:** prove the thesis mechanically.
- **Build:** a test/example that adds a brand-new trivial architecture **and** a
  trivial new stage using only the public protocols.
- **Done when:** both run end-to-end through the existing runner and a front end
  with **zero edits** to `runner.py`, existing stages, or existing architectures.
- **Depends on:** W6, W12 (meaningful once M3–M5 exist)
