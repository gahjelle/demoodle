# LLM Demo Tool — Vision & End Goal

> A pedagogical, runnable tool that builds up a language model from the simplest
> possible thing to a real (tiny) transformer with post-training, **training for
> real on a normal laptop**, with every stage inspectable and every seam swappable.

---

## 1. Purpose

This is a teaching instrument, not a production framework. Its job is to make the
internals of modern LLMs **legible** by letting a presenter (or a self-guided
learner) move along three orthogonal axes one step at a time and *watch what
changes*. Nothing is faked: every model genuinely trains, every number shown is
real, and everything runs on CPU (or modest Apple-Silicon/GPU) in seconds to a
few minutes.

The design follows **functional core / imperative shell**:
- **Core** (`core/`, `stages/`) is pure and deterministic: data in, data out,
  RNG passed explicitly as a value. No disk, clock, or printing.
- **Shell** (`shell/`) owns all I/O, RNG sourcing, caching, and rendering.
- **Front ends** (`frontends/`) are interchangeable views over the same shell.

---

## 2. The Three Axes

The whole demo is structured so a presenter can change exactly one thing and
show the effect. The axes are independent.

### Axis A — Architecture (model capacity)
1. **Bigram (count-based)** — a V×V frequency table. No training, just counting.
   Teaches: "a model is a next-token probability distribution."
2. **Bigram (learned)** — a single V×V weight matrix trained by gradient descent.
   Teaches: gradients, loss, optimization — the bigram from #1 but *learned*.
3. **MLP (Bengio-style)** — character embeddings + a context window + hidden layer.
   Teaches: embeddings, fixed context, nonlinearity.
4. **Transformer (1–2 layers, 1–2 heads)** — the real thing, tiny.
   Teaches: self-attention, positional information, residual stream.

All four implement the same `Architecture`/`Inspectable` protocol, so the front
end swaps them with a single selection.

### Axis B — Tokenization (input representation)
1. **Char-level** — trivial `{char: id}` map. Vocab ~30–90.
2. **Byte-level BPE** — merge rules *learned* from the corpus.

The tokenizer is a first-class **trained artifact** produced by a stage, not a
fixed config. Swapping char→BPE resizes the embedding table and shortens
sequences but touches no model or runner code.

### Axis C — Training / Post-training (behavior shaping)
1. **Pretrain** — next-token prediction on a corpus.
2. **SFT (supervised fine-tuning)** — train on curated (prompt → desired output)
   pairs to shape format/behavior.
3. **Reward model (RM)** — train a scorer on preference pairs (A better than B).
4. **RLHF / preference optimization** — optimize the policy against the reward
   (PPO-style) or directly against preferences (DPO-style), with a frozen
   reference model for KL control.

These compose as a **dependency graph**, not a linear sequence (RM and the SFT
policy are both *inputs* to RLHF).

#### Where "human feedback" actually lives
RLHF is widely misunderstood, so the demo makes the data flow explicit:

```
humans → preference labels → reward model (TRAINED) → policy (TRAINED via PPO)
```

- Humans only ever **label preferences** (pick the better of two completions).
  This is the sole point where a human is in the loop.
- Those labels **train the reward model** — this is the human feedback becoming a
  trainable artifact. After this, the humans are gone; their judgment lives as RM
  weights.
- PPO then optimizes the policy against the **reward model**, not against humans.
  No humans are present in the RL loop.

So there are **two training steps and one human-labeling step**; humans never
touch the RL loop directly. The **frozen reference** is a *separate* mechanism — a
fixed copy of the SFT policy used as a KL anchor (a leash against drift), not a
reward signal. Reference = leash; reward model = goal.

**DPO differs:** it skips the reward model and trains the policy *directly* on the
preference pairs (one training step, no RM artifact). Same human feedback;
different plumbing.

#### Decision: build both, default to DPO
- **DPO is the default `rlhf` stage** — stable, fast, demo-safe, genuinely current.
- **PPO is a labeled variant** (`rlhf_ppo`) for the full classical-pipeline story
  and the reward-vs-KL dashboard. Same `needs=("sft_policy", "reward_model")`.
- The reward-model stage is built regardless: it's a worthwhile visualization on
  its own and PPO requires it. DPO simply doesn't consume it.
- **`Policy` carries an optional, nullable value head** from day one (`None`
  except during a PPO run). This one cheap slot avoids retrofitting the
  architecture classes later.

---

## 3. Core Concepts & Data Model

### Artifacts
Everything produced or consumed is an immutable, named **Artifact** (a tagged
union):
- `Corpus` — raw training text.
- `Tokenizer` — encode/decode + vocab (char map or learned BPE merges).
- `Dataset` — tokenized, batched training data derived from a corpus + tokenizer.
- `Policy` — a trained model (architecture + weights). The thing that generates.
  Carries an **optional, nullable value head** (`None` except during PPO).
- `RewardModel` — a trained scorer: `(sequence) -> scalar`.
- `PreferenceData` — pairs of (preferred, rejected) completions. May be produced
  automatically by a toy reward function **or interactively by a human** (see §4).
- `Metrics` — loss curves, eval numbers, trajectories.

Artifacts live in a dict keyed by name. Multiple models can coexist (this is
essential for RLHF: policy + frozen reference + reward model are *separate*
named artifacts).

### Stages
A **Stage** is a pure transform that declares its dependencies by name:

```python
@dataclass(frozen=True)
class Stage:
    name: str
    needs: tuple[str, ...]      # e.g. ("sft_policy", "reward_model")
    produces: str               # e.g. "rlhf_policy"
    run: Callable[[Mapping[str, Artifact], RNG], Artifact]   # pure
```

The shell **topologically sorts** the declared stages and executes them, caching
each artifact by a hash of (stage code + inputs + config + seed). This makes
RLHF's dependency structure explicit and visible rather than hidden inside a
linear fold.

Example end-state graph:

```
corpus ─┬─> train_tokenizer ─> tokenizer ─┐
        │                                  ├─> build_dataset ─> dataset ─> pretrain ─> base_policy
        └──────────────────────────────────┘                                              │
                                                                                           ├─> sft ─> sft_policy ──┐
preference_data ─> train_reward_model ─> reward_model ─────────────────────────────────────┤        │             │
                                                                                            └────────┴─> rlhf ─> rlhf_policy
                                                                                              (needs sft_policy + reward_model;
                                                                                               reads sft_policy as frozen ref)
```

### Protocols (the swap points)

```python
class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    @property
    def vocab_size(self) -> int: ...

class Architecture(Protocol):
    def init_state(self, cfg: Config, rng: RNG) -> Policy: ...
    def forward(self, policy: Policy, tokens: Seq) -> Output: ...   # logits

class Inspectable(Protocol):
    def call(self, prompt: Seq, temperature: float) -> Output: ...  # universal:
        # next-token distribution + sampling. Every architecture has this.
    def explain(self, prompt: Seq) -> dict[str, Any]: ...           # optional:
        # arch-specific internals (attention maps, embeddings). Feature-detected.

class Frontend(Protocol):
    # renders the shell's outputs; CLI / web / notebook all implement this.
    ...
```

`call` is universal; `explain` is best-effort. A bigram has no attention map and
simply omits it; a transformer fills it in. Front ends feature-detect.

### RNG
RNG is a small **value type** with `.split()` (JAX-style), threaded explicitly.
No global seeding anywhere in core. This is what makes the core deterministic and
snapshot-testable.

---

## 4. Data Sources (all public, tiny, committed to the repo)

- **Names corpus** (~32K US baby names, public-domain census data, a few hundred
  KB). The starter. A bigram already produces plausible fake names, so even the
  simplest model *visibly works*.
- **TinyShakespeare** (~1MB, single public-domain text file). For the transformer
  and BPE stages; output is recognizably structured.

Both are committed as plain text — no download step, no API, no licensing
footnote, runs offline in a live talk.

### Toy reward tasks (for honest RLHF without a GPU farm)
Because the models are char-level/tiny, post-training stays *real* by choosing
tasks with a **computable reward**, e.g.:
- "prefer names starting with a vowel"
- "prefer shorter names"
- "prefer outputs that end with a period"

The reward function is a few lines, fully real, and the behavior shift is obvious
in `call` output. Preference data for the RM can be generated by sampling the base
model and labeling pairs with the reward function — so the RM learns a *real*
(if simple) signal.

### Live human-in-the-loop labeling (the RLHF "aha")
The toy reward function stands in for thousands of human labels, but the demo can
also put a **real human in the loop** for a handful of pairs:

1. A front end samples completions from the policy and shows the human two at a
   time: "which is better?" The human clicks. Repeat ~20–40 times (seconds).
2. Those clicks become a `PreferenceData` artifact (built in the **shell** — the
   human interaction never enters the pure core; only the resulting data does).
3. The **existing, unchanged** reward-model/DPO stages consume it. The tiny RM
   retrains in seconds; the policy's behavior visibly shifts toward the human's
   taste.

**Honest framing baked into the demo:** ~30 labels genuinely moves a tiny RM, and
that *is* RLHF in miniature — but the output is noisy, and the demo says so
explicitly ("real systems use tens of thousands of labels; watch how unstable 30
is"). Best mode is a **blend**: seed with auto-generated pairs from the toy reward,
let the human add or override a few, and show their fingerprint on the result.
This is the same structure as production RLHF, which is what makes it honest
teaching material.

---

## 5. What Each Stage Demonstrates (the payoff)

| Stage | The "aha" |
|---|---|
| Count bigram | A model is just P(next \| current). Sampling = generation. |
| Learned bigram | Same table, but reached by gradient descent. Watch loss fall. |
| MLP | Context and embeddings let it look back further. |
| Transformer | Attention; `explain()` renders which positions attend where. |
| Char→BPE | Merges build vocab live; same prompt re-segments; sequences shorten; the "count the r's" failure becomes explainable. |
| Pretrain→SFT | Format/behavior can be shaped with a few curated examples. |
| Reward model | Preferences become a learned scalar; visualize RM scores. |
| RLHF/DPO | Policy drifts toward reward while KL to the frozen reference holds it back. Show reward ↑ and KL, side by side. |

---

## 6. Front Ends (interchangeable)

All consume the same shell API; none touch core.
- **CLI** — train a graph, run `call` with temperature, dump metrics. Day-one target.
- **Notebook** — same calls, inline plots (loss curves, attention heatmaps, BPE merges).
- **Web** — interactive: pick architecture/tokenizer/training axis, type a prompt,
  watch token-by-token probabilities and (where available) attention. The
  "showpiece" front end.

---

## 7. Inspection & Visualization (cross-cutting)

The thing that makes this a *demo* and not just a trainer:
- **Token-by-token generation** with the probability distribution at each step,
  and a temperature slider showing how it sharpens/flattens.
- **Loss curves** for any training stage (real numbers).
- **Attention heatmaps** via `explain()` for transformers.
- **BPE merge timeline** — vocab building itself, step by step.
- **Tokenization comparison** — same string under char vs BPE, with token counts.
- **RLHF dashboard** — reward and KL-to-reference over training steps.
- **Live preference labeling** — a two-completions-side-by-side picker that turns
  human clicks into `PreferenceData`, then shows the resulting behavior shift.

---

## 8. Non-Goals

- Not production-scale; not GPU-cluster training; not competitive quality.
- No live training that takes longer than a coffee break; slow artifacts are
  **cached**, never **faked**.
- No simulate/compute split — everything shown is genuinely computed.

---

## 9. Definition of Done (the full target)

A user can, from any front end:
1. Choose a point on each of the three axes (architecture × tokenizer × training).
2. Trigger a real training run (or hit cache) on a committed corpus, on a laptop,
   within minutes.
3. Call the resulting model interactively and inspect its internals.
4. Add a *new* architecture, tokenizer, or training stage by implementing a
   protocol/stage **without modifying the runner or other stages** — proving the
   plug-and-play thesis.
