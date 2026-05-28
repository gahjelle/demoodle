"""Plain CLI entry point for the demoo command."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import configaroo
import cyclopts
import torch
from rich.console import Console
from rich.live import Live
from rich.text import Text

from demoodle.architectures.bigram import BigramArchitecture
from demoodle.architectures.trigram import TrigramArchitecture
from demoodle.config import config
from demoodle.core.rng import RNG
from demoodle.data.loaders import load_corpus
from demoodle.data.stages import make_build_dataset_stage
from demoodle.shell import runner
from demoodle.tokenizers.char import CharTokenizer, make_train_tokenizer_stage
from demoodle.training.stages import make_pretrain_stage

if TYPE_CHECKING:
    from collections.abc import Callable

    from demoodle.core.types import Artifact, Policy, TrainingMetrics
    from demoodle.ports.protocols import ArchitectureProtocol, InspectableProtocol

app = cyclopts.App(name="demoo", help="Demoo — build your own (large) language models")
stdout = Console()

_SPARK = "⠀⣀⣤⣶⣿"
_SEED = 280177


def _sparkline(losses: list[float], width: int, n_steps: int) -> str:
    """Return a braille sparkline string of `width` chars encoding `losses`.

    Bars are normalized to [initial_loss, 0]: full bar = initial loss, empty = zero.
    Filled cells grow left-to-right; unfilled cells remain '⠀' (progress indicator).
    """
    cells = [_SPARK[0]] * width
    if not losses:
        return "".join(cells)
    initial = losses[0]
    steps_per_cell = max(1, n_steps // width)
    for i in range(width):
        start = i * steps_per_cell
        if start >= len(losses):
            break
        end = min(start + steps_per_cell, len(losses))
        window = losses[start:end]
        window_loss = sum(window) / len(window)
        level = min(1.0, max(0.0, window_loss / initial)) if initial > 0 else 0.0
        cells[i] = _SPARK[round(level * 4)]
    return "".join(cells)


def _make_display(losses: list[float], step: int, n_steps: int, width: int) -> Text:
    """Return a two-line Rich Text with step/loss header and live sparkline."""
    pad = len(str(n_steps))
    header = f"Training  step {step + 1:>{pad}}/{n_steps} • loss: {losses[-1]:.4f}"
    return Text(f"{header}\n{_sparkline(losses, width, n_steps)}")


def _make_arch(vocab_size: int) -> ArchitectureProtocol:
    """Instantiate the active architecture with the given vocab size."""
    match config.architecture.active:
        case "bigram":
            return BigramArchitecture(vocab_size=vocab_size)
        case "trigram":
            return TrigramArchitecture(vocab_size=vocab_size)
        case other:
            msg = f"Unknown architecture {other!r} — is it implemented yet?"
            raise ValueError(msg)


def _run_pipeline(
    rng: RNG,
    cache_dir: Path,
    on_step: Callable[[int, float], None] | None = None,
) -> dict[str, Artifact]:
    """Run the full day-one pipeline, returning all artifacts.

    Two-phase: phase 1 resolves vocab_size from the tokenizer stage; phase 2
    runs the full graph with an arch that knows its vocabulary size.
    """
    corpus = load_corpus(config.corpus.active)
    tok_stage = make_train_tokenizer_stage()

    phase1 = runner.run([tok_stage], {"corpus": corpus}, rng, cache_dir)
    vocab_size = cast("CharTokenizer", phase1["tokenizer"]).vocab_size

    arch = _make_arch(vocab_size)
    dataset_stage = make_build_dataset_stage()
    pretrain_stage = make_pretrain_stage(
        arch, config.training.pretrain, on_step=on_step
    )

    return runner.run(
        [tok_stage, dataset_stage, pretrain_stage],
        {"corpus": corpus},
        rng,
        cache_dir,
    )


def _generate(
    arch: InspectableProtocol,
    policy: Policy,
    tokenizer: CharTokenizer,
    rng: RNG,
    temperature: float,
    prompt: str,
    context_length: int,
    max_len: int = 100,
) -> str:
    """Sample a single continuation from `prompt`, stopping on newline or max_len."""
    token_ids = tokenizer.encode(prompt)
    prompt_len = len(token_ids)
    bos_id = tokenizer.encode("\n")[0]
    for _ in range(max_len):
        rng, step_rng = rng.split()
        padded = [bos_id] * max(0, context_length - len(token_ids)) + token_ids
        seq = torch.tensor(padded[-context_length:])
        output = arch.call(seq, policy, step_rng, temperature)
        if output.sampled_ids is None:
            msg = "arch.call() did not return sampled_ids"
            raise RuntimeError(msg)
        next_id = int(output.sampled_ids.item())
        token_ids.append(next_id)
        if tokenizer.decode([next_id]) == "\n":
            break
    return tokenizer.decode(token_ids[prompt_len:]).rstrip("\n")


@app.command
def show_config(section: str | None = None) -> None:
    """Show the Demoodle configuration."""
    configaroo.print_configuration(config, section=section)


@app.command
def train() -> None:
    """Train the day-one model and display a live braille loss curve."""
    cache_dir = Path(config.paths.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = RNG(seed=_SEED)
    n_steps = config.training.pretrain.n_steps
    width = max(40, stdout.width - 20)

    losses: list[float] = []

    with Live(console=stdout, refresh_per_second=4, transient=True) as live:

        def on_step(step: int, loss: float) -> None:
            losses.append(loss)
            live.update(_make_display(losses, step, n_steps, width))

        artifacts = _run_pipeline(rng, cache_dir, on_step=on_step)

    metrics = cast("TrainingMetrics", artifacts["metrics"])
    final_losses = losses or metrics.losses
    label = "Done  " if losses else "Cached"
    stdout.print(f"{label}  step {n_steps}/{n_steps} • loss: {final_losses[-1]:.4f}")
    stdout.print(_sparkline(final_losses, width, n_steps))


@app.command
def call(
    *,
    temperature: float = 1.0,
    n: int = 5,
    prompt: str = "\n",
) -> None:
    """Generate text continuations from the trained policy."""
    cache_dir = Path(config.paths.cache_dir)
    rng = RNG(seed=_SEED)
    train()
    artifacts = _run_pipeline(rng, cache_dir)

    tokenizer = cast("CharTokenizer", artifacts["tokenizer"])
    policy = cast("Policy", artifacts["base_policy"])
    arch = _make_arch(tokenizer.vocab_size)
    arch_gen = cast("InspectableProtocol", arch)
    context_length = arch.context_length

    display_prefix = "" if prompt == "\n" else prompt
    for _ in range(n):
        rng, gen_rng = rng.split()
        result = _generate(
            arch_gen, policy, tokenizer, gen_rng, temperature, prompt, context_length
        )
        stdout.print(f"{display_prefix}{result}")
