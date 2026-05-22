"""Pipeline stages for the data layer."""

from typing import TYPE_CHECKING, cast

import torch

from demoodle.core.types import Dataset
from demoodle.ports.protocols import Stage

if TYPE_CHECKING:
    from demoodle.core.rng import RNG
    from demoodle.core.types import Artifact, Corpus
    from demoodle.ports.protocols import TokenizerProtocol


def make_build_dataset_stage(config_hash: str = "") -> Stage:
    """Return a Stage that encodes a Corpus into a Dataset using a tokenizer."""

    def run(artifacts: dict[str, Artifact], _rng: RNG) -> dict[str, Artifact]:
        corpus = cast("Corpus", artifacts["corpus"])
        tokenizer = cast("TokenizerProtocol", artifacts["tokenizer"])
        text = corpus.text if corpus.text.endswith("\n") else corpus.text + "\n"
        ids = tokenizer.encode(text)
        return {"dataset": Dataset(tokens=torch.tensor(ids, dtype=torch.long))}

    return Stage(
        name="build_dataset",
        needs=["corpus", "tokenizer"],
        produces=["dataset"],
        config_hash=config_hash,
        run=run,
    )
