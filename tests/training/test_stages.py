"""Tests for training pipeline stages."""

from typing import TYPE_CHECKING

import torch

from demoodle.architectures.bigram import BigramArchitecture
from demoodle.config.schemas import PretrainConfig
from demoodle.core.rng import RNG
from demoodle.core.types import Dataset, Policy, TrainingMetrics
from demoodle.ports.protocols import Stage
from demoodle.shell import runner
from demoodle.training.stages import make_pretrain_stage

if TYPE_CHECKING:
    from pathlib import Path

    from demoodle.core.types import Artifact

VOCAB = 10
RNG0 = RNG(seed=0)

_FAST_CONFIG = PretrainConfig(learning_rate=0.1, batch_size=16, n_steps=200)
_CONVERGENCE_CONFIG = PretrainConfig(learning_rate=0.1, batch_size=32, n_steps=500)


def _make_dataset(size: int = 500) -> Dataset:
    tokens = torch.randint(0, VOCAB, (size,))
    return Dataset(tokens=tokens)


@torch.no_grad()
def _weights_equal(p1: Policy, p2: Policy) -> bool:
    sd1 = p1.model.state_dict()
    sd2 = p2.model.state_dict()
    return all(torch.equal(sd1[k], sd2[k]) for k in sd1)


# ---------------------------------------------------------------------------
# Artifact types
# ---------------------------------------------------------------------------


def test_stage_produces_correct_artifact_types() -> None:
    arch = BigramArchitecture(vocab_size=VOCAB)
    stage = make_pretrain_stage(arch, _FAST_CONFIG)
    result = stage.run({"dataset": _make_dataset()}, RNG0)
    assert isinstance(result["base_policy"], Policy)
    assert isinstance(result["metrics"], TrainingMetrics)


def test_metrics_length_equals_n_steps() -> None:
    arch = BigramArchitecture(vocab_size=VOCAB)
    stage = make_pretrain_stage(arch, _FAST_CONFIG)
    result = stage.run({"dataset": _make_dataset()}, RNG0)
    metrics = result["metrics"]
    assert isinstance(metrics, TrainingMetrics)
    assert len(metrics.losses) == _FAST_CONFIG.n_steps


# ---------------------------------------------------------------------------
# Loss decreases
# ---------------------------------------------------------------------------


def test_loss_decreases_over_training() -> None:
    arch = BigramArchitecture(vocab_size=VOCAB)
    stage = make_pretrain_stage(arch, _CONVERGENCE_CONFIG)
    result = stage.run({"dataset": _make_dataset(2000)}, RNG0)
    metrics = result["metrics"]
    assert isinstance(metrics, TrainingMetrics)
    tenth = len(metrics.losses) // 10
    first_mean = sum(metrics.losses[:tenth]) / tenth
    last_mean = sum(metrics.losses[-tenth:]) / tenth
    assert last_mean < first_mean


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_stage_run_is_deterministic_under_fixed_rng() -> None:
    arch = BigramArchitecture(vocab_size=VOCAB)
    stage = make_pretrain_stage(arch, _FAST_CONFIG)
    dataset = _make_dataset()
    r1 = stage.run({"dataset": dataset}, RNG0)
    r2 = stage.run({"dataset": dataset}, RNG0)
    assert isinstance(r1["base_policy"], Policy)
    assert isinstance(r2["base_policy"], Policy)
    assert _weights_equal(r1["base_policy"], r2["base_policy"])


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------


def test_config_hash_differs_on_learning_rate_change() -> None:
    arch = BigramArchitecture(vocab_size=VOCAB)
    s1 = make_pretrain_stage(
        arch, PretrainConfig(learning_rate=0.1, batch_size=16, n_steps=100)
    )
    s2 = make_pretrain_stage(
        arch, PretrainConfig(learning_rate=0.01, batch_size=16, n_steps=100)
    )
    assert s1.config_hash != s2.config_hash


def test_config_hash_differs_on_vocab_size_change() -> None:
    s1 = make_pretrain_stage(BigramArchitecture(vocab_size=10), _FAST_CONFIG)
    s2 = make_pretrain_stage(BigramArchitecture(vocab_size=20), _FAST_CONFIG)
    assert s1.config_hash != s2.config_hash


def test_config_hash_differs_on_context_length_change() -> None:
    s1 = make_pretrain_stage(
        BigramArchitecture(vocab_size=VOCAB, context_length=1), _FAST_CONFIG
    )
    s2 = make_pretrain_stage(
        BigramArchitecture(vocab_size=VOCAB, context_length=2), _FAST_CONFIG
    )
    assert s1.config_hash != s2.config_hash


# ---------------------------------------------------------------------------
# Runner / cache integration
# ---------------------------------------------------------------------------


def test_cache_hit_on_rerun(tmp_path: Path) -> None:
    calls: list[int] = []
    arch = BigramArchitecture(vocab_size=VOCAB)
    base_stage = make_pretrain_stage(arch, _FAST_CONFIG)

    def counting_run(artifacts: dict[str, Artifact], rng: RNG) -> dict[str, Artifact]:
        calls.append(1)
        return base_stage.run(artifacts, rng)

    stage = Stage(
        name=base_stage.name,
        needs=base_stage.needs,
        produces=base_stage.produces,
        config_hash=base_stage.config_hash,
        run=counting_run,
    )
    initial: dict[str, Artifact] = {"dataset": _make_dataset()}
    rng = RNG(seed=42)
    runner.run(stages=[stage], initial_artifacts=initial, rng=rng, cache_dir=tmp_path)
    runner.run(stages=[stage], initial_artifacts=initial, rng=rng, cache_dir=tmp_path)
    assert len(calls) == 1


def test_different_rng_seed_produces_cache_miss(tmp_path: Path) -> None:
    calls: list[int] = []
    arch = BigramArchitecture(vocab_size=VOCAB)
    base_stage = make_pretrain_stage(arch, _FAST_CONFIG)

    def counting_run(artifacts: dict[str, Artifact], rng: RNG) -> dict[str, Artifact]:
        calls.append(1)
        return base_stage.run(artifacts, rng)

    stage = Stage(
        name=base_stage.name,
        needs=base_stage.needs,
        produces=base_stage.produces,
        config_hash=base_stage.config_hash,
        run=counting_run,
    )
    initial: dict[str, Artifact] = {"dataset": _make_dataset()}
    runner.run(
        stages=[stage], initial_artifacts=initial, rng=RNG(seed=1), cache_dir=tmp_path
    )
    runner.run(
        stages=[stage], initial_artifacts=initial, rng=RNG(seed=2), cache_dir=tmp_path
    )
    assert len(calls) == 2
