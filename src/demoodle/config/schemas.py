"""Pydantic schema definitions for demoodle configuration."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base model: immutable and strict (no extra fields allowed)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


#
# Architecture
#
class BigramConfig(StrictModel):
    """Bigram architecture config — vocab_size comes from the tokenizer artifact."""


class MLPConfig(StrictModel):
    """Bengio-style MLP architecture config."""

    embedding_dim: int
    context_length: int
    hidden_size: int


class TransformerConfig(StrictModel):
    """Tiny transformer architecture config."""

    embedding_dim: int
    context_length: int
    n_layers: int
    n_heads: int
    dropout: float


class ArchitecturesConfig(StrictModel):
    """All architecture sub-configs plus the active selection key."""

    active: str
    bigram: BigramConfig
    mlp: MLPConfig
    transformer: TransformerConfig


#
# Tokenizer sub-configs
#
class CharConfig(StrictModel):
    """Character-level tokenizer config — vocabulary is derived from the corpus."""


class BPEConfig(StrictModel):
    """Byte-level BPE tokenizer config."""

    vocab_size: int


#
# Section configs
#
class TokenizersConfig(StrictModel):
    """All tokenizer sub-configs plus the active selection key."""

    active: str
    char: CharConfig
    bpe: BPEConfig


class PretrainConfig(StrictModel):
    """Hyperparameters for the pretrain stage."""

    learning_rate: float
    batch_size: int
    n_steps: int


class TrainingConfig(StrictModel):
    """Per-stage training hyperparameters."""

    pretrain: PretrainConfig


class CorpusEntryConfig(StrictModel):
    """Metadata for a single corpus."""

    url: str
    description: str
    license: str


class CorpusConfig(StrictModel):
    """All corpus entries plus the active selection key."""

    active: str
    names: CorpusEntryConfig
    shakespeare: CorpusEntryConfig
    code: CorpusEntryConfig


class PathsConfig(StrictModel):
    """Filesystem paths used by the shell layer."""

    cache_dir: str


#
# Top-level config
#
class DemoodleConfig(StrictModel):
    """Root configuration model for demoodle."""

    architecture: ArchitecturesConfig
    tokenizer: TokenizersConfig
    training: TrainingConfig
    corpus: CorpusConfig
    paths: PathsConfig
