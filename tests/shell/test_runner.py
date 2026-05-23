"""Tests for shell/runner.py — topological stage executor."""

import warnings
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from demoodle.core.rng import RNG
from demoodle.core.types import Artifact, Corpus
from demoodle.ports import Stage
from demoodle.shell.runner import run

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stage(
    name: str,
    needs: list[str],
    produces: list[str],
    config_hash: str = "",
    calls: list[str] | None = None,
    seeds: list[int] | None = None,
) -> Stage:
    """Build a test Stage that emits a Corpus for each key it produces."""

    def _run(_artifacts: dict[str, Artifact], rng: RNG) -> dict[str, Artifact]:
        if calls is not None:
            calls.append(name)
        if seeds is not None:
            seeds.append(rng.seed)
        return {k: Corpus(text=k) for k in produces}

    return Stage(
        name=name, needs=needs, produces=produces, config_hash=config_hash, run=_run
    )


def _rng(seed: int = 0) -> RNG:
    return RNG(seed=seed)


# ---------------------------------------------------------------------------
# 0. Dirty-worktree warning
# ---------------------------------------------------------------------------


def test_dirty_worktree_emits_user_warning(tmp_path: Path) -> None:
    s = _stage("s", needs=[], produces=["out"])
    with (
        patch("demoodle.shell.persistence.WORKTREE_DIRTY", new=True),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        run([s], {}, _rng(), tmp_path)
    assert any(
        issubclass(w.category, UserWarning)
        and "uncommitted" in str(w.message)
        and str(tmp_path) in str(w.message)
        for w in caught
    )


def test_clean_worktree_emits_no_warning(tmp_path: Path) -> None:
    s = _stage("s", needs=[], produces=["out"])
    with (
        patch("demoodle.shell.persistence.WORKTREE_DIRTY", new=False),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        run([s], {}, _rng(), tmp_path)
    assert not any(issubclass(w.category, UserWarning) for w in caught)


# ---------------------------------------------------------------------------
# 1. Graph validation
# ---------------------------------------------------------------------------


def test_duplicate_produces_raises_naming_both_stages(tmp_path: Path) -> None:
    s1 = _stage("alpha", needs=[], produces=["x"])
    s2 = _stage("beta", needs=[], produces=["x"])
    with pytest.raises(ValueError, match="alpha") as exc_info:
        run([s1, s2], {}, _rng(), tmp_path)
    assert "beta" in str(exc_info.value)
    assert "x" in str(exc_info.value)


def test_cycle_raises_with_unsatisfied_stage_names(tmp_path: Path) -> None:
    # a needs "y" (produced by b), b needs "x" (produced by a) — mutual deadlock
    s_a = _stage("a", needs=["y"], produces=["x"])
    s_b = _stage("b", needs=["x"], produces=["y"])
    with pytest.raises(ValueError, match="unresolvable") as exc_info:
        run([s_a, s_b], {}, _rng(), tmp_path)
    msg = str(exc_info.value)
    assert "a" in msg
    assert "b" in msg


def test_missing_initial_artifact_raises_naming_stage_and_artifact(
    tmp_path: Path,
) -> None:
    s = _stage("lonely", needs=["ghost"], produces=["out"])
    with pytest.raises(ValueError, match="unresolvable") as exc_info:
        run([s], {}, _rng(), tmp_path)
    msg = str(exc_info.value)
    assert "lonely" in msg
    assert "ghost" in msg


# ---------------------------------------------------------------------------
# 2. Topological sort
# ---------------------------------------------------------------------------


def test_stages_run_in_dependency_order(tmp_path: Path) -> None:
    calls: list[str] = []
    s_a = _stage("a", needs=[], produces=["x"], calls=calls)
    s_b = _stage("b", needs=["x"], produces=["y"], calls=calls)
    s_c = _stage("c", needs=["y"], produces=["z"], calls=calls)
    # Pass in reverse order — runner must sort them correctly
    run([s_c, s_b, s_a], {}, _rng(), tmp_path)
    assert calls == ["a", "b", "c"]


def test_independent_stages_within_tier_sorted_by_name(tmp_path: Path) -> None:
    calls: list[str] = []
    s_z = _stage("z", needs=[], produces=["z_art"], calls=calls)
    s_a = _stage("a", needs=[], produces=["a_art"], calls=calls)
    s_m = _stage("m", needs=[], produces=["m_art"], calls=calls)
    run([s_z, s_a, s_m], {}, _rng(), tmp_path)
    assert calls == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# 3. Core runner behaviour
# ---------------------------------------------------------------------------


def test_run_passes_only_declared_needs_to_stage(tmp_path: Path) -> None:
    received: dict[str, Artifact] = {}

    def capturing_run(artifacts: dict[str, Artifact], _rng: RNG) -> dict[str, Artifact]:
        received.update(artifacts)
        return {"out": Corpus(text="out")}

    stage = Stage(
        name="s", needs=["needed"], produces=["out"], config_hash="", run=capturing_run
    )
    initial: dict[str, Artifact] = {
        "needed": Corpus(text="needed"),
        "extra": Corpus(text="extra"),
    }
    run([stage], initial, _rng(), tmp_path)
    assert set(received.keys()) == {"needed"}


def test_run_returns_initial_and_produced_artifacts(tmp_path: Path) -> None:
    s = _stage("s", needs=["x"], produces=["y"])
    initial: dict[str, Artifact] = {"x": Corpus(text="x")}
    result = run([s], initial, _rng(), tmp_path)
    assert "x" in result
    assert "y" in result


def test_rng_split_is_unconditional_before_cache_check(tmp_path: Path) -> None:
    """Stage 2 receives the same RNG seed whether stage 1 is cached or runs fresh."""
    seeds: list[int] = []

    s1 = _stage("a", needs=[], produces=["mid"])
    s2_v1 = _stage("b", needs=["mid"], produces=["out"], config_hash="v1", seeds=seeds)
    s2_v2 = _stage("b", needs=["mid"], produces=["out"], config_hash="v2", seeds=seeds)

    root = _rng(seed=7)

    # Run 1: both stages run fresh (s2_v1)
    run([s1, s2_v1], {}, root, tmp_path)

    # Run 2: s1 hits cache; s2_v2 (different config_hash) misses — same seed expected
    run([s1, s2_v2], {}, root, tmp_path)

    assert len(seeds) == 2
    assert seeds[0] == seeds[1]


# ---------------------------------------------------------------------------
# 4. Cache integration
# ---------------------------------------------------------------------------


def test_second_identical_run_calls_no_stage_run(tmp_path: Path) -> None:
    calls: list[str] = []
    s = _stage("s", needs=[], produces=["out"], calls=calls)
    run([s], {}, _rng(), tmp_path)
    run([s], {}, _rng(), tmp_path)
    assert len(calls) == 1


def test_different_seed_causes_reexecution(tmp_path: Path) -> None:
    calls: list[str] = []
    s = _stage("s", needs=[], produces=["out"], calls=calls)
    run([s], {}, _rng(seed=1), tmp_path)
    run([s], {}, _rng(seed=2), tmp_path)
    assert len(calls) == 2


def test_three_stage_pipeline_end_to_end(tmp_path: Path) -> None:
    """Full pipeline: corpus → tokenizer → dataset, with cache verification."""
    calls: list[str] = []
    s_a = _stage("corpus", needs=[], produces=["corpus"], calls=calls)
    s_b = _stage("tokenize", needs=["corpus"], produces=["tokenizer"], calls=calls)
    s_c = _stage(
        "dataset", needs=["corpus", "tokenizer"], produces=["dataset"], calls=calls
    )

    root = _rng(seed=99)
    result = run([s_a, s_b, s_c], {}, root, tmp_path)

    # corpus → tokenize → dataset is the only valid order
    assert calls == ["corpus", "tokenize", "dataset"]
    # Both initial-equivalent and produced artifacts present
    assert {"corpus", "tokenizer", "dataset"} == set(result.keys())

    calls.clear()
    run([s_a, s_b, s_c], {}, root, tmp_path)
    # Second run: all cached
    assert calls == []
