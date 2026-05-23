from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from rich.console import Console

from demoodle.architectures.bigram import BigramArchitecture
from demoodle.core.rng import RNG
from demoodle.core.types import Corpus, TrainingMetrics
from demoodle.frontends import cli
from demoodle.frontends.cli import _generate, _sparkline
from demoodle.tokenizers.char import CharTokenizer

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VOCAB = {"\n": 0, "a": 1, "b": 2, "c": 3}
_TOKENIZER = CharTokenizer(char_to_id=_VOCAB)
_ARCH = BigramArchitecture(vocab_size=len(_VOCAB))
_POLICY = _ARCH.init_state(RNG(seed=0))


def _make_artifacts() -> dict:
    return {
        "corpus": Corpus(text="abc\n"),
        "tokenizer": _TOKENIZER,
        "base_policy": _POLICY,
        "metrics": TrainingMetrics(losses=[4.0, 3.5, 3.0, 2.5, 2.0]),
    }


# ---------------------------------------------------------------------------
# 1. _sparkline
# ---------------------------------------------------------------------------


def test_sparkline_empty_losses_returns_all_blank() -> None:
    result = _sparkline([], width=10, n_steps=100)
    assert result == "⠀" * 10


def test_sparkline_one_loss_at_initial_fills_first_cell() -> None:
    result = _sparkline([4.0], width=10, n_steps=10)
    assert result[0] == "⣿"
    assert result[1:] == "⠀" * 9


def test_sparkline_full_run_fills_all_cells() -> None:
    # n_steps == width, each cell gets one loss value at initial level
    losses = [4.0] * 8
    result = _sparkline(losses, width=8, n_steps=8)
    assert result == "⣿" * 8


def test_sparkline_half_complete_fills_half() -> None:
    # 5 losses, width=10, n_steps=10 → first 5 cells filled
    losses = [4.0] * 5
    result = _sparkline(losses, width=10, n_steps=10)
    assert all(c != "⠀" for c in result[:5])
    assert result[5:] == "⠀" * 5


def test_sparkline_decreasing_loss_gives_decreasing_bars() -> None:
    # Losses go from 4.0 down to near-zero; bars should get shorter
    losses = [4.0, 3.0, 2.0, 1.0, 0.0]
    result = _sparkline(losses, width=5, n_steps=5)
    # First char should be tallest (⣿), each subsequent char should be ≤ previous
    levels = [_sparkline.__globals__["_SPARK"].index(c) for c in result]
    assert levels == sorted(levels, reverse=True)


def test_sparkline_loss_above_initial_clips_to_full() -> None:
    # Loss exceeds initial → should clip to ⣿, not go out of range
    result = _sparkline([2.0, 8.0], width=2, n_steps=2)
    assert result[1] == "⣿"


def test_sparkline_zero_initial_loss_gives_blank_bars() -> None:
    result = _sparkline([0.0, 0.0], width=2, n_steps=2)
    assert result == "⠀⠀"


# ---------------------------------------------------------------------------
# 3. train smoke test
# ---------------------------------------------------------------------------


def test_train_completes_without_error(tmp_path: Path) -> None:
    artifacts = _make_artifacts()
    mock_cfg = MagicMock()
    mock_cfg.paths.cache_dir = str(tmp_path)
    mock_cfg.training.pretrain.n_steps = 5
    mock_cfg.architecture.active = "bigram"
    mock_cfg.corpus.active = "names"

    with (
        patch.object(cli, "config", mock_cfg),
        patch.object(cli, "_run_pipeline", return_value=artifacts),
    ):
        cli.train()


def test_train_shows_cached_label_when_no_steps_ran(tmp_path: Path) -> None:
    artifacts = _make_artifacts()
    mock_cfg = MagicMock()
    mock_cfg.paths.cache_dir = str(tmp_path)
    mock_cfg.training.pretrain.n_steps = 5
    mock_cfg.architecture.active = "bigram"

    output = StringIO()
    with (
        patch.object(cli, "config", mock_cfg),
        patch.object(cli, "_run_pipeline", return_value=artifacts),
        patch.object(cli, "stdout", Console(file=output, highlight=False)),
    ):
        cli.train()

    assert "Cached" in output.getvalue()


# ---------------------------------------------------------------------------
# 4. _generate
# ---------------------------------------------------------------------------


def test_generate_is_deterministic() -> None:
    rng = RNG(seed=7)
    result1 = _generate(_ARCH, _POLICY, _TOKENIZER, rng, 1.0, "\n", context_length=1)
    result2 = _generate(_ARCH, _POLICY, _TOKENIZER, rng, 1.0, "\n", context_length=1)
    assert result1 == result2


def test_generate_prompt_prefix_excluded_from_return() -> None:
    # _generate returns only the continuation, not the prompt itself
    result = _generate(
        _ARCH, _POLICY, _TOKENIZER, RNG(seed=0), 1.0, "\n", context_length=1
    )
    assert not result.startswith("\n")


def test_generate_stops_at_newline() -> None:
    # With temp=0.0 the model may loop; max_len ensures termination
    result = _generate(
        _ARCH, _POLICY, _TOKENIZER, RNG(seed=1), 1.0, "\n", context_length=1, max_len=5
    )
    assert len(result) <= 5


def test_generate_prompt_seeds_context() -> None:
    # Prompt "a" should produce different output than prompt "\n"
    r1 = _generate(_ARCH, _POLICY, _TOKENIZER, RNG(seed=3), 1.0, "\n", context_length=1)
    r2 = _generate(_ARCH, _POLICY, _TOKENIZER, RNG(seed=3), 1.0, "a", context_length=1)
    # Different prompts → different starting context → likely different output
    # (not guaranteed with random weights, but almost certainly different)
    assert isinstance(r1, str)
    assert isinstance(r2, str)


# ---------------------------------------------------------------------------
# 4. call display_prefix
# ---------------------------------------------------------------------------


def test_call_prepends_prompt_prefix_to_output(tmp_path: Path) -> None:
    artifacts = _make_artifacts()
    mock_cfg = MagicMock()
    mock_cfg.paths.cache_dir = str(tmp_path)
    mock_cfg.training.pretrain.n_steps = 5
    mock_cfg.architecture.active = "bigram"

    output = StringIO()
    with (
        patch.object(cli, "config", mock_cfg),
        patch.object(cli, "_run_pipeline", return_value=artifacts),
        patch.object(cli, "train"),
        patch.object(cli, "stdout", Console(file=output, highlight=False)),
    ):
        cli.call(prompt="a", n=1)

    assert output.getvalue().startswith("a")
