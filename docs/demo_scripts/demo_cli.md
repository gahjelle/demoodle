# Demoodle — CLI Demo Script & Design

> Companion to DEMO_TUI.md. Same five-act arc, same thesis (architecture ×
> tokenizer × training = three independent knobs), but reconceived for the command
> line. The CLI is **not** a degraded TUI — it's a different instrument with its own
> strengths. Where the TUI is a *live dashboard you play*, the CLI is a *transcript
> you produce*: every move is a typed command with durable, scrollable, copy-pasteable
> output. That makes it the better tool for some beats and the worse tool for others.

---

## 1. What the CLI does *better* than the TUI

- **The command itself is the explanation.** `demoodle gen --arch transformer --tok bpe`
  is self-documenting; the audience sees exactly which knob moved. No hidden UI state.
- **Durable scrollback.** Output stays on screen; you can scroll back to the bigram
  gibberish while showing transformer output. The transcript *is* the before/after.
- **Diffs as first-class text.** `demoodle compare A B` printing a side-by-side or a
  unified diff is crisp and screenshot-friendly — arguably clearer than a split pane.
- **Copy-paste / reproducibility.** Every command can be pasted into a README, a
  slide, or an issue. The demo doubles as documentation.
- **Scriptability.** The whole arc can be a shell script (`demo.sh`) you step through
  with Enter, or pipe to `asciinema`. Trivial to rehearse and to share async.
- **Honest "show me the seed."** Flags like `--seed 1337` make determinism explicit
  and on-screen, reinforcing "this is reproducible, not a magic trick."

## 2. What the TUI does better (so the CLI should *not* try to)

- **Live continuous controls** (the temperature slider, watching a loss curve draw in
  real time, an attention heatmap you scrub). The CLI fakes these with discrete
  snapshots — fine, but don't pretend it's smooth.
- **Simultaneous panels.** The TUI shows graph + generation + inspect at once; the CLI
  shows one thing per command. Lean into sequencing instead of cramming.
- **The live human-labeling "aha"** is more visceral in the TUI/web. The CLI version
  works (prompt y/n in the terminal) but is the one beat where you may want to switch
  to the TUI/web if available.

**Design rule:** the CLI demo trades *liveness* for *legibility and durability*.
Discrete, named, reproducible commands with clean text output beat animation here.

---

## 3. CLI surface the demo assumes

A consistent verb set, every command accepting the four axis flags
(`--arch --tok --corpus --train`) plus `--seed`:

```
demoodle gen      --arch --tok --corpus --train [--prompt P] [--temp T] [--top-k K] [--top-p P] [-n N]
demoodle compare  <axis> <from> <to> [other flags]   # holds all but <axis> fixed
demoodle tokens   <string> --tok char|bpe [--corpus C]   # segmentation + count
demoodle merges   --corpus C [--top N] [--animate]       # BPE merge timeline
demoodle attn     --prompt P [--arch transformer]        # attention as text grid
demoodle eval     --corpus C [--train STAGE] [-n N]      # W21 validity gauge
demoodle label    [--blend] [-n N]                       # interactive preference y/n
demoodle train    <stage> [flags] [--live]               # run/retrain, show loss
demoodle graph    [flags]                                # print the DAG + cache hits
demoodle plug     <module>                               # register a new arch/stage
```

Two cross-cutting flags drive the demo narrative:
- `--compare-to <preset>`: any `gen`/`eval` can print itself *next to* a saved
  baseline, so before/after is one command.
- `--explain`: append a one-line plain-English note of what changed vs the last
  invocation (e.g. "+ self-attention: can look at any earlier token"). This is the
  CLI's version of the TUI title card.

Output is **text-first but `--json`-able** everywhere (same W28 API underneath), so
the demo is pretty by default and pipeable when needed.

---

## 4. The acts (CLI staging)

Same arc as the TUI; the *medium-specific staging* is what differs. Each beat = one
command (shown verbatim) + the point to make.

### Prologue — "Three knobs and a coin flip"
```
demoodle gen --arch bigram --tok char --corpus names --train pretrain --temp 0.8
demoodle gen ... --temp 0.1      # collapses
demoodle gen ... --temp 2.0      # noise
```
- CLI strength: run the *same command* three times changing only `--temp`; the three
  outputs stack in scrollback so the audience reads the distribution sharpening/melting
  down the screen. The flag value *is* the knob.
- Print per-token probabilities as a static bar table for the first token (`--bars`):
  legible and screenshot-able where the TUI's live bars aren't durable.

### Act I — Architecture: "Give it memory" (corpus = names)
```
demoodle compare arch bigram mlp --corpus names --tok char --explain
demoodle compare arch mlp transformer --corpus names --tok char --explain
demoodle attn --prompt "emm" --arch transformer
```
- CLI strength: `compare` prints from→to samples and the loss delta as a clean block —
  the durable diff. `--explain` supplies the one-liner.
- Climax: `attn` renders the attention matrix as a **text heatmap** (shaded blocks /
  numbers). Less dazzling than the TUI grid, but it's right there in the transcript and
  you can point at specific cells. *This is the beat where TUI wins on wow; CLI wins on
  "here's the actual matrix."*

### Act II — Tokenization: "How it reads" (corpus = code)
```
demoodle tokens "strawberry" --tok char
demoodle tokens "strawberry" --tok bpe --corpus code      # few opaque chunks
demoodle merges --corpus code --top 20                    # learned merges, static list
demoodle merges --corpus code --animate                   # step-by-step (optional)
demoodle gen --arch transformer --tok bpe --corpus code --temp 0.7
demoodle eval --corpus code --train pretrain -n 50        # validity %
```
- **This act is the CLI's strongest.** Tokenization is inherently textual: segmentation,
  token counts, and merge lists are *born* as terminal output. `tokens` printing the
  same string under char vs bpe with counts is clearer than any panel.
- `merges --top 20` as a static ranked list (with the corpus's signature merges:
  4-space indent, `def `, `self.`, `):`) is the headline — and trivially comparable by
  rerunning with `--corpus names`/`shakespeare` and scrolling between them.
- `eval` prints the W21 syntax-validity gauge as a number + bar; sets up Act III.

### Act III — Training / Post-training: "Teach it what we want"
```
demoodle compare train pretrain sft --arch transformer --tok bpe --corpus code --explain
demoodle eval --train sft -n 50            # validity ticks up vs pretrain
demoodle label --blend -n 25               # interactive: prints A/B, you type 1 or 2
demoodle train dpo --live                  # loss draws as it trains (the one live beat)
demoodle compare train sft dpo --explain   # behavior shift
demoodle eval --train dpo -n 50            # gauge rises
demoodle train ppo --live                  # reward up, KL tracked (printed per step)
```
- CLI strength: the **eval gauge as a number across stages** tells the post-training
  story quantitatively — `pretrain 34% → sft 51% → dpo 68%` is a one-line punchline you
  can't get as cleanly in the TUI. Consider a `demoodle eval --across pretrain,sft,dpo`
  that prints exactly that progression table.
- The `label` beat: terminal y/n works and is honest, but flag this as the moment to
  *switch to TUI/web if present* for the visceral version. With `--blend` it seeds from
  auto-pairs so 25 clicks suffice.
- `train --live` is the CLI's one genuinely-live beat: loss/reward/KL printed per step
  as it computes (node not cached). Everything else is pre-warmed and instant.

### Finale — "The whole machine"
```
demoodle compare full \
  --from "bigram char names pretrain" \
  --to   "transformer bpe code dpo"          # the whole journey, one block
demoodle plug examples/silly_arch.py          # register a new architecture live
demoodle gen --arch silly --tok bpe --corpus code   # runs through the same graph, zero edits
demoodle graph                                # show DAG + all-cache-hits
```
- CLI strength for the closer: `compare full` prints the Prologue gibberish directly
  above the final competent output — the entire arc in one screenshot.
- `plug` + immediately `gen --arch silly` is the **mechanical proof (W31)**: a new arch
  registered and run through the unchanged runner, on screen, in two commands. This is
  arguably *more* convincing in the CLI than the TUI because there's visibly no UI
  wiring — just a new flag value that works.
- `graph` prints the DAG with cache-hit markers (✓ cached / * fresh) as an ASCII tree,
  closing on "every stage is just a node; you only ever recompute the diff."

---

## 5. Wow-moment index (CLI)

| #   | Act      | Moment                                        | CLI vs TUI                       |
| --- | -------- | --------------------------------------------- | -------------------------------- |
| 1   | Prologue | `--temp` stacked outputs down the scrollback  | CLI ~ (durable, less live)       |
| 2   | I        | `compare arch` from→to block + loss delta     | CLI = (cleaner diff)             |
| 3   | I        | `attn` text heatmap                           | TUI wins (CLI shows real matrix) |
| 4   | II       | `tokens` char vs bpe segmentation + counts    | **CLI wins**                     |
| 5   | II       | `merges --top 20` ranked learned merges       | **CLI wins**                     |
| 6   | II       | `eval` validity gauge                         | CLI =                            |
| 7   | III      | `label` → `train dpo --live` → behavior shift | TUI/web wins                     |
| 8   | III      | `eval --across` progression table 34→51→68%   | **CLI wins**                     |
| 9   | Finale   | `compare full` whole-arc block                | CLI = (screenshot-friendly)      |
| 10  | Finale   | `plug` + `gen` new arch in two commands       | **CLI wins**                     |

Irreducible CLI spine if short on time: **4 → 5 → 8 → 10** (note this differs from the
TUI's 3→4→7→10 — the CLI leans on tokenization and the eval progression, the TUI on
attention and the live human loop).

---

## 6. Robustness (CLI)

- **`demoodle demo warmup`** (shared with the TUI harness, W32) pre-materializes every
  config so all but `train --live` are instant.
- **`demo.sh` stepper:** the full arc as a commented shell script; press Enter between
  commands. Rehearsable, shareable, and recordable with `asciinema`.
- **Determinism on screen:** `--seed` visible in every command; rehearsed outputs
  reproduce verbatim.
- **No-color / wide-terminal flags** for projectors; `--json` escape hatch if a render
  misbehaves.

---

## 7. How this lands against the current plans

Mostly clean — the CLI is W12 (early) but the demo *script* leans on capabilities that
only exist after the axes are built, which is fine (the script is a Finale-era artifact).
A few gaps to fold into later work items:

- **W12 (CLI front end)** currently does just `train` + `call --temperature`. The demo
  needs the richer verb set (`compare`, `tokens`, `merges`, `attn`, `eval`, `label`,
  `graph`, `plug`). **Don't bloat W12** — keep it the day-one slice. Instead these verbs
  are naturally added by the items that introduce each capability (e.g. `tokens`/`merges`
  with W20, `eval` with W21, `label` with W24, `attn` with W17). Recommend adding a
  one-line "CLI: expose `<verb>`" to the **done-when** of W17, W20, W21, W24, and W31,
  so the CLI grows alongside each feature instead of in a big lump.
- **W28** already promises streaming token distributions, `compare(A,B)`, and cache
  hit/miss — the CLI's `--bars`, `compare`, and `graph` are just text renderings of
  those. No change needed; the CLI is a thin renderer over the same API as the TUI.
- **`eval --across <stages>`** (the progression table) is a small, high-value CLI-only
  view. Suggest noting it in **W21**'s done-when ("CLI can print a cross-stage
  progression table").
- **`compare` as a CLI verb** maps onto W28's `compare()` helper — add "CLI exposes a
  `compare` verb over this" to W28's done-when.
- **W32 (Demo harness)** should explicitly cover *both* front ends: warmup is shared;
  add the `demo.sh` stepper + asciinema-friendliness as CLI-specific items in its build.

None of this reopens the architecture. The pattern holds: every front-end verb is a
renderer over the W28 shell API, and the CLI vs TUI split is purely presentation.
