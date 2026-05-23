"""Content-addressed artifact cache for the demoodle pipeline."""

import hashlib
import subprocess
from importlib.metadata import version
from typing import TYPE_CHECKING

import torch

from demoodle.core.types import Corpus, Dataset, Policy, TrainingMetrics
from demoodle.tokenizers.char import CharTokenizer

if TYPE_CHECKING:
    from pathlib import Path

    from demoodle.core.rng import RNG
    from demoodle.core.types import Artifact
    from demoodle.ports.protocols import Stage

_DEMOODLE_VERSION: str = version("demoodle")


def _git_id() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except FileNotFoundError, subprocess.CalledProcessError:
        return ""


def is_worktree_dirty() -> bool:
    """Return True if the git working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except FileNotFoundError, subprocess.CalledProcessError:
        return False


_GIT_ID: str = _git_id()
WORKTREE_DIRTY: bool = is_worktree_dirty()


def _hash_artifact(artifact: Artifact) -> str:
    h = hashlib.sha256()
    match artifact:
        case CharTokenizer(char_to_id=char_to_id):
            h.update(repr(sorted(char_to_id.items())).encode())
        case Corpus(text=text):
            h.update(text.encode())
        case TrainingMetrics(losses=losses):
            h.update(repr(losses).encode())
        case Dataset(tokens=tokens):
            h.update(
                bytes(tokens.cpu().contiguous().view(torch.uint8).flatten().tolist())
            )
        case Policy(model=model, value_head=value_head):
            for name, param in sorted(model.state_dict().items()):
                h.update(name.encode())
                h.update(
                    bytes(param.cpu().contiguous().view(torch.uint8).flatten().tolist())
                )
            if value_head is not None:
                for name, param in sorted(value_head.state_dict().items()):
                    h.update(name.encode())
                    h.update(
                        bytes(
                            param.cpu()
                            .contiguous()
                            .view(torch.uint8)
                            .flatten()
                            .tolist()
                        )
                    )
    return h.hexdigest()


def cache_key(stage: Stage, inputs: dict[str, Artifact], rng: RNG) -> str:
    """Return a stable hex key identifying this exact stage invocation."""
    h = hashlib.sha256()
    h.update(stage.name.encode())
    h.update(_DEMOODLE_VERSION.encode())
    h.update(_GIT_ID.encode())
    h.update(stage.config_hash.encode())
    for artifact_name in sorted(inputs):
        h.update(artifact_name.encode())
        h.update(_hash_artifact(inputs[artifact_name]).encode())
    h.update(str(rng.seed).encode())
    return h.hexdigest()


def save(key: str, artifacts: dict[str, Artifact], cache_dir: Path) -> None:
    """Persist an artifact dict to the cache directory under the given key."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts, cache_dir / f"{key}.pt")


def load(key: str, cache_dir: Path) -> dict[str, Artifact] | None:
    """Load a cached artifact dict, or return None on a cache miss."""
    path = cache_dir / f"{key}.pt"
    if not path.exists():
        return None
    # weights_only=False: artifact dicts contain dataclasses, not just tensors.
    return torch.load(path, weights_only=False)  # type: ignore[return-value]
