"""Topological stage executor with content-addressed caching."""

from typing import TYPE_CHECKING

from demoodle.shell import persistence

if TYPE_CHECKING:
    from pathlib import Path

    from demoodle.core.rng import RNG
    from demoodle.core.types import Artifact
    from demoodle.ports.protocols import Stage


def _validate_no_duplicate_produces(stages: list[Stage]) -> None:
    seen: dict[str, str] = {}
    for stage in stages:
        for artifact in stage.produces:
            if artifact in seen:
                msg = (
                    f"Artifact {artifact!r} is produced by both"
                    f" {seen[artifact]!r} and {stage.name!r}"
                )
                raise ValueError(msg)
            seen[artifact] = stage.name


def _topo_sort(stages: list[Stage], initial: frozenset[str]) -> list[Stage]:
    available = initial
    remaining = list(stages)
    order: list[Stage] = []

    while remaining:
        ready = sorted(
            [s for s in remaining if all(n in available for n in s.needs)],
            key=lambda s: s.name,
        )
        if not ready:
            unsatisfied = {
                s.name: [n for n in s.needs if n not in available] for s in remaining
            }
            msg = (
                f"Stage graph has unresolvable dependencies"
                f" (cycle or missing artifacts): {unsatisfied}"
            )
            raise ValueError(msg)
        order = order + ready
        available = available | {name for s in ready for name in s.produces}
        remaining = [s for s in remaining if s not in ready]

    return order


def run(
    stages: list[Stage],
    initial_artifacts: dict[str, Artifact],
    rng: RNG,
    cache_dir: Path,
) -> dict[str, Artifact]:
    """Execute stages in topological order, using the cache when possible."""
    _validate_no_duplicate_produces(stages)
    sorted_stages = _topo_sort(stages, frozenset(initial_artifacts))

    artifacts: dict[str, Artifact] = dict(initial_artifacts)

    for stage in sorted_stages:
        stage_rng, rng = rng.split()
        inputs = {k: artifacts[k] for k in stage.needs}
        key = persistence.cache_key(stage, inputs, stage_rng)

        cached = persistence.load(key, cache_dir)
        if cached is not None:
            outputs = cached
        else:
            outputs = stage.run(inputs, stage_rng)
            persistence.save(key, outputs, cache_dir)

        artifacts = {**artifacts, **outputs}

    return artifacts
