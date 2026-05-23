"""Training pipeline stages."""

import hashlib
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F  # noqa: N812

from demoodle.core.types import Dataset, TrainingMetrics
from demoodle.ports.protocols import Stage

if TYPE_CHECKING:
    from demoodle.config.schemas import PretrainConfig
    from demoodle.core.rng import RNG
    from demoodle.core.types import Artifact
    from demoodle.ports.protocols import ArchitectureProtocol


def _config_hash(arch: ArchitectureProtocol, config: PretrainConfig) -> str:
    h = hashlib.sha256()
    h.update(config.model_dump_json().encode())
    h.update(str(arch.vocab_size).encode())
    h.update(str(arch.context_length).encode())
    return h.hexdigest()


def make_pretrain_stage(arch: ArchitectureProtocol, config: PretrainConfig) -> Stage:
    """Return a Stage that trains `arch` on a dataset.

    Produces ``base_policy`` and ``metrics`` artifacts.
    """

    def run(artifacts: dict[str, Artifact], rng: RNG) -> dict[str, Artifact]:
        dataset = cast("Dataset", artifacts["dataset"])
        tokens = dataset.tokens
        n = len(tokens)
        context_len = arch.context_length

        policy = arch.init_state(rng)
        optimizer = torch.optim.Adam(  # type: ignore[attr-defined]
            policy.model.parameters(), lr=config.learning_rate
        )

        generator = rng.generator()
        losses: list[float] = []

        for _ in range(config.n_steps):
            offsets = torch.randint(
                n - context_len, (config.batch_size,), generator=generator
            )
            inputs = torch.stack([tokens[o : o + context_len] for o in offsets])
            targets = tokens[offsets + context_len]

            logits = policy.model(inputs.squeeze(1) if context_len == 1 else inputs)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        return {
            "base_policy": policy,
            "metrics": TrainingMetrics(losses=losses),
        }

    return Stage(
        name="pretrain",
        needs=["dataset"],
        produces=["base_policy", "metrics"],
        config_hash=_config_hash(arch, config),
        run=run,
    )
