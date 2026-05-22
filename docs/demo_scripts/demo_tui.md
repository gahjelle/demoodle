# Demoodle — TUI Demo Script & Design

> Companion to PLANS.md / VISION.md. Describes the Textual TUI front end (W29) as a
> *live teaching instrument*: not a dashboard you read, but an instrument you play.
> The thesis the UI must make physical: an LLM is **three independent knobs**
> (architecture × tokenizer × training), wired through one artifact graph, and you
> can turn any one knob and watch the effect in isolation.

---

## 1. The core mechanic

Everything in the demo reduces to one loop:

1. **A baseline is fixed** along all axes but one.
2. The presenter **turns the active knob one notch** (single keystroke).
3. The runner re-executes only the affected stages — **the cache makes this instant**
   (pre-warmed) — and the TUI shows a **before → after split**.
4. The presenter **lingers on one wow detail**, then advances.

Each beat is ~60–90s and is fully described by: *title card (one line) + keystroke +
what appears in the split + the detail to point at*. That keeps the pace brisk and
makes every change self-explaining.

The unit of the demo is the **diff**, never the raw run.

---

## 2. TUI layout

```
┌ BASELINE ──────────────────────────────────────────────────────────────────┐
│  Arch: [Transformer]   Tok: [BPE]   Corpus: [code]   Train: [DPO]          │
│         ▲ active axis (bright)   others locked (dim)        seed: 1337     │
├──────────────┬───────────────────────────────────────┬─────────────────────┤
│ STAGE GRAPH  │  STAGE (generation)                   │  INSPECT (context)  │
│              │                                       │                     │
│ corpus ✓     │  prompt> def fib(n):                  │  loss curve  ▁▂▃▅▆▇ │
│  └tok  ✓     │  ┌ token-by-token ───────────────┐    │                     │
│   └data ✓    │  │ return ████████ 0.62          │    │  attention heatmap  │
│    └pre  ✓   │  │ fib    ███      0.18          │    │  ░░▒▒▓▓██ (grid)    │
│     └sft ✓   │  │ n      ██       0.11          │    │                     │
│      └dpo ◐  │  └───────────────────────────────┘    │  eval: valid 71% ▲  │
│  (◐ = live)  │  temperature ●━━━━━━○  0.8            │  reward ▁▃▅  KL ▂▂▃ │
├──────────────┴───────────────────────────────────────┴─────────────────────┤
│ WHAT CHANGED:  + context window (MLP→Transformer)   loss 2.41→1.78  ✓cached│
└────────────────────────────────────────────────────────────────────────────┘
```

- **Baseline bar (top):** the four-dimensional config. Active axis is bright; locked
  axes are dimmed. Changing the active dimension is `←/→` (or number keys). This bar
  is the spine of the whole demo — it's how the audience always knows "what is being
  held still and what is moving."
- **Stage graph (left):** the live DAG. Nodes light up as they run; **cached nodes
  vs freshly-trained nodes are color-coded** (e.g. dim ✓ = cache hit, pulsing ◐ =
  training now). This is the modularity thesis made visible — adding a knob re-runs
  only the downstream sub-tree.
- **Stage / generation (center):** a prompt box, streamed token-by-token output with
  **per-token probability bars**, and the **temperature slider** (live, `↑/↓`). The
  tactile heart of the demo.
- **Inspect panel (right, context-sensitive):** swaps with the active axis —
  loss-curve sparkline always; attention heatmap when a transformer is loaded;
  merge timeline + char-vs-BPE segmentation in the tokenization act; reward/KL traces
  + RM score in the post-training act; the W21 generation-quality gauge throughout.
- **"What changed" ribbon (bottom):** the one-line label of the last move + the
  before→after deltas (loss Δ, eval Δ) + a cache badge. This is the title card.

A first-class **compare mode** (`c`) splits the center pane into two configs side by
side — the killer view for every wow moment (gibberish | structured, char | BPE,
base | DPO).

---

## 3. The "restart" model

Two distinct restarts, both first-class in the UI:

**Soft reset (within an act).** Step back one notch to re-show a before/after, or nudge
the active knob. Always cached, always instant. Bound to `[` / `]` (prev/next notch on
the active axis) and `space` (re-roll a sample at the current config).

**Axis switch (between acts).** The structural restart. One action:
`promote current config to baseline, then unlock a new axis and re-lock the rest`.
Visually: three locks close, one opens. Bound to a single key per axis (`1` arch /
`2` tok / `3` train). Two narrative flavors, chosen per act:

- **Isolate** (Acts I–III): when entering an act, reset the *other* axes to the weak
  baseline so the new axis's contribution is shown cleanly and independently
  ("this is what tokenization *alone* buys you"). Best for teaching.
- **Stack** (Finale): keep every prior gain and show cumulative improvement
  ("look how far we've come"). Best for the closer.

Because the runner is an artifact graph (not a linear fold), an axis switch never
forces a full retrain — it rehydrates the cached sub-tree for the new baseline and
only re-runs what genuinely differs. **The cache is what makes "restart" feel like
flipping a switch instead of rebooting.**

---

## 4. The acts

### Prologue — "Three knobs and a coin flip" (~3 min)
**Goal:** install the mental model and make sampling tactile before any improvement.

- Open on the baseline bar showing all three knobs, then dim two and leave the third —
  narrate "we'll turn exactly one at a time."
- Load the **weakest** config: bigram · char · names · pretrain-only. Generate.
  Output: vowel-soup pseudo-names. *Honest framing: "this is a coin-weighted by the
  previous letter — nothing more."*
- **Interactive beat:** grab the **temperature slider**. Drag to 0.1 → output collapses
  to the same few letters; drag to 2.0 → output dissolves into noise. Per-token
  probability bars sharpen/flatten live.
- *Wow:* the audience feels that "the model" is just a probability distribution they
  can physically squeeze.

### Act I — Architecture: "Give it memory" (corpus = names) (~6 min)
**Thesis card:** *Same data, same tokenizer. Only the model changes.*

- **Beat 1 — Bigram → MLP.** Keystroke `→`. Card: "+ embeddings + a context window."
  Compare-mode split: names get more name-like; loss drops on the sparkline.
- **Beat 2 — MLP → Transformer.** Keystroke `→`. Card: "+ self-attention: it can look
  at *any* earlier character." Split: clearly structured names; loss drops again.
- **Beat 3 — Attention reveal (climax).** Switch inspect panel to the **attention
  heatmap**. Type a prompt; watch the grid light up where the model "looks back."
  *Wow:* the abstract word "attention" becomes a glowing diagonal you can point at.
- *Restart out:* promote Transformer to baseline; isolate-reset corpus/training.

### Act II — Tokenization: "How it reads" (corpus = code) (~7 min)
**Thesis card:** *Same model, same training. Only the tokenizer changes.*

- **Beat 1 — The famous failure.** With **char** tokenizer, ask it to "count the
  letters in strawberry" (or just show segmentation). Then switch to **BPE** and show
  the same string chunked into a few opaque tokens — *the model literally cannot see
  the letters.* Side-by-side token counts.
- **Beat 2 — Merge timeline (climax 1).** Animate BPE merges forming on the code
  corpus: watch `' '+' '→'  '`, then `def`, `self.`, `):` emerge. *Wow:* tokenization
  is *learned*, and on code it learns indentation and keywords — visibly different
  from the names/Shakespeare merges (flip the corpus knob to compare top-merges).
- **Beat 3 — Code generation + the gauge (climax 2).** Generate Python. It looks like
  code — plausible indentation, keywords, `def`/`return`. The **W21 syntax-validity
  gauge** shows a real number (e.g. ~30–60%). *Honest framing: a toy CPU model won't
  emit runnable code, but the shape is unmistakable — and that gauge is about to
  matter.*
- *Restart out:* keep Transformer · BPE; isolate-reset training to pretrain-only.

### Act III — Training / Post-training: "Teach it what we want" (~10 min)
**Thesis card:** *Same model, same tokenizer. Only the training changes.* This is the
longest act and holds the biggest wow.

- **Beat 1 — Pretrain → SFT.** Card: "show it ~a few hundred curated examples of the
  format we want." Split: output snaps toward the target format; the validity gauge
  ticks up.
- **Beat 2 — Auto-preferences + reward model.** Card: "a rule scores pairs (shorter /
  valid / period-ending); a small reward model learns that rule." Inspect panel shows
  RM scoring sampled pairs preferred > rejected.
- **Beat 3 — Human-in-the-loop (THE wow, and it MUST be live).** Drop into the
  two-completions picker. The presenter (or audience by show-of-hands) clicks "which
  is better?" ~25–30 times. Then **retrain live** — the loss curve draws in real time
  (node pulses ◐ in the graph, *not* a cache hit). Re-generate: behavior visibly
  shifts toward what the room just chose. *Wow:* "your clicks just became weights."
  Surface the honest caveat: ~30 labels is noisy; real systems use tens of thousands.
- **Beat 4 — DPO vs PPO (climax).** Show DPO's behavior shift with the **KL-to-
  reference** trace, then flip to PPO and watch the **reward rising while KL is
  tracked** on the live dashboard. Card: "two roads to the same goal; one uses the
  reward model, one doesn't."
- *Restart out:* nothing reset — we're about to stack.

### Finale — "The whole machine" (~4 min)
**Goal:** cumulative payoff + prove the architecture claim mechanically.

- **Stack montage:** in compare mode, put the Prologue's bigram-char-pretrain gibberish
  next to the final Transformer-BPE-DPO output. One slider scrubs the journey. *Wow:*
  the full arc in a single screen.
- **Plug-and-play proof (W31), live:** with the audience watching, register a brand-new
  trivial architecture (or stage) and watch it appear as a new notch on the baseline
  bar and run through the *exact same* graph and UI — **zero edits** to the runner.
  *Wow:* the thesis isn't a slide, it's a property you just exercised on stage.
- Close on the three knobs, all lit: "architecture, tokenization, training — that's the
  whole machine."

---

## 5. Wow-moment index (for cutting to time)

| #   | Act      | Moment                                     | Live or cached   |
| --- | -------- | ------------------------------------------ | ---------------- |
| 1   | Prologue | Temperature squeezes the distribution      | live (instant)   |
| 2   | I        | Bigram → Transformer sample transformation | cached           |
| 3   | I        | Attention heatmap lights up                | cached           |
| 4   | II       | "Count the letters" failure, char vs BPE   | cached           |
| 5   | II       | Merge timeline animates on code            | cached           |
| 6   | II       | Code generation + validity gauge           | cached           |
| 7   | III      | Human clicks → retrain → behavior shift    | **must be live** |
| 8   | III      | PPO reward rising while KL tracked         | cached or live   |
| 9   | Finale   | Stack montage (gibberish → competent)      | cached           |
| 10  | Finale   | New architecture plugged in live           | **must be live** |

If short on time, the irreducible spine is **3 → 4 → 7 → 10** (one per act).

---

## 6. Live-demo robustness

The instrument is only fun if it never stalls. Build these in:

- **Pre-warm command** (`demoodle demo warmup`): materializes every config the script
  visits so all transitions except the two intentionally-live beats are cache hits. The
  TUI's ✓ badges let you *honestly say* "precomputed" while beats 7 and 10 prove it's
  real by training before their eyes.
- **Determinism:** fixed seed shown in the baseline bar; rehearsed samples reproduce
  exactly, so you know what will appear. The human-labeling beat is the one place you
  embrace nondeterminism — that's the point.
- **Scripted path mode:** `n` advances to the next *pre-planned* beat (title card +
  config + prompt all set), so you can drive the whole arc with one finger. `g <act>`
  jumps to any act if time runs short.
- **Fallback samples:** a known-good cached sample per beat, so even if a live retrain
  wobbles you can fall back to a rehearsed result without breaking flow.
- **Panic key:** `r` re-rolls the current beat from cache; `b` snaps the baseline back
  to the last promoted config (undo a fat-fingered axis change).

---

## 7. What this implies for the build (hooks back into PLANS.md)

The TUI (W29) should stay a pure consumer of the W28 shell API — but the demo wants a
few capabilities worth making explicit when W28/W29 land:

- **W28 additions:** a streaming generation API (yields per-token distributions, not
  just the final string) for the probability bars; a `compare(configA, configB)` helper
  returning aligned before/after data; surfacing the W21 gauge and the live RLHF
  reward/KL traces as incremental updates.
- **W29 sub-features:** the baseline-bar state machine (lock/unlock per axis, promote-
  to-baseline), compare-mode split, the DAG widget with per-node cache/running state,
  and scripted-path/warmup/fallback plumbing from §6.
- **Cache visibility:** the runner already keys on (stage + code + inputs + config +
  seed); expose "was this a hit?" per node so the graph can color it. Small W5/W6
  addition, big demo payoff.
- **Optional new item:** consider a `W32. Demo harness` (warmup + scripted path +
  fallback cache) so the presentation polish doesn't leak into core or W29 widget code.

None of this changes the artifact-graph thesis — it *depends* on it. The cache and the
single-axis-at-a-time discipline are exactly what make the restarts feel like flipping
a switch.
