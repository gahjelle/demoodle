"""Character-level tokenizer and its training stage."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from demoodle.ports.protocols import Stage

if TYPE_CHECKING:
    from demoodle.core.rng import RNG
    from demoodle.core.types import Artifact, Corpus


@dataclass(frozen=True)
class CharTokenizer:
    """One token per character; vocabulary built from a corpus."""

    char_to_id: dict[str, int]

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self.char_to_id)

    def encode(self, text: str) -> list[int]:
        """Map each character to its token id."""
        return [self.char_to_id[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        """Map token ids back to a string."""
        id_to_char = {v: k for k, v in self.char_to_id.items()}
        return "".join(id_to_char[i] for i in ids)


def make_train_tokenizer_stage(config_hash: str = "") -> Stage:
    """Return a Stage that builds a CharTokenizer from a corpus artifact."""

    def run(artifacts: dict[str, Artifact], _rng: RNG) -> dict[str, Artifact]:
        corpus: Corpus = artifacts["corpus"]  # ty: ignore[invalid-assignment]
        char_to_id = {c: i for i, c in enumerate(sorted(set(corpus.text)))}
        return {"tokenizer": CharTokenizer(char_to_id=char_to_id)}

    return Stage(
        name="train_tokenizer",
        needs=["corpus"],
        produces=["tokenizer"],
        config_hash=config_hash,
        run=run,
    )
