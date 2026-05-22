import importlib
import sys

import pytest
from pydantic import ValidationError

import demoodle.config as _cfg_module
from demoodle.config import config
from demoodle.config.schemas import (
    ArchitecturesConfig,
    BigramConfig,
    CorpusConfig,
    CorpusEntryConfig,
    DemoodleConfig,
    MLPConfig,
    PathsConfig,
    TokenizersConfig,
    TrainingConfig,
)


def test_config_is_demoodle_config() -> None:
    assert isinstance(config, DemoodleConfig)


def test_config_default_architecture() -> None:
    assert config.architecture.active == "bigram"


def test_config_default_sections() -> None:
    assert isinstance(config.architecture, ArchitecturesConfig)
    assert isinstance(config.tokenizer, TokenizersConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.corpus, CorpusConfig)
    assert isinstance(config.paths, PathsConfig)


def test_config_corpus_names_metadata() -> None:
    names = config.corpus.names
    assert isinstance(names, CorpusEntryConfig)
    assert names.url
    assert names.description
    assert names.license


def test_config_architecture_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMOODLE_ARCHITECTURE", "mlp")
    # Force reload since config is a module-level singleton
    importlib.reload(_cfg_module)
    assert _cfg_module.config.architecture.active == "mlp"
    # Restore original state for other tests
    monkeypatch.delenv("DEMOODLE_ARCHITECTURE", raising=False)
    importlib.reload(_cfg_module)


def test_config_mlp_section_is_typed() -> None:
    mlp = config.architecture.mlp
    assert isinstance(mlp, MLPConfig)
    assert isinstance(mlp.embedding_dim, int)
    assert isinstance(mlp.context_length, int)
    assert isinstance(mlp.hidden_size, int)


def test_config_bigram_section_exists() -> None:
    assert isinstance(config.architecture.bigram, BigramConfig)


def test_config_active_key_resolves_to_sub_config() -> None:
    for active in ("bigram", "mlp", "transformer"):
        arch_cfg = config.architecture.model_copy(update={"active": active})
        sub = getattr(arch_cfg, active)
        assert sub is not None


def test_config_is_immutable() -> None:
    with pytest.raises(ValidationError):
        config.architecture = config.architecture  # type: ignore[misc]


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DemoodleConfig.model_validate({"unknown_field": "value"})


# Clean up any cached module state that might bleed between tests
def teardown_module() -> None:
    if "demoodle.config" in sys.modules:
        importlib.reload(sys.modules["demoodle.config"])
