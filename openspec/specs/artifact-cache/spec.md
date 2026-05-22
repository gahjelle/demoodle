# Spec: Artifact Cache

## Purpose

Provides content-addressed caching for pipeline stage outputs. A stable cache key is derived from the stage identity, code version, configuration, inputs, and RNG seed; artifacts are serialised to disk and restored on cache hit, allowing repeated runs to skip expensive computation.

## Requirements

### Requirement: cache_key produces a stable, unique key for a stage invocation
The system SHALL provide a `cache_key(stage, inputs, rng)` function in `demoodle.shell.persistence` that returns a hex string uniquely identifying the combination of stage name, code version (`__version__` + git short ID), `stage.config_hash`, sorted input artifact hashes, and `rng.seed`. The same inputs SHALL always produce the same key; any change to any component SHALL produce a different key.

#### Scenario: Same inputs produce the same key
- **WHEN** `cache_key` is called twice with identical stage, inputs, and rng
- **THEN** both calls return the same string

#### Scenario: Different seed produces a different key
- **WHEN** `cache_key` is called with two RNGs that differ only in seed
- **THEN** the returned keys are different

#### Scenario: Different config_hash produces a different key
- **WHEN** `cache_key` is called with two Stage instances that differ only in `config_hash`
- **THEN** the returned keys are different

#### Scenario: Different input artifact produces a different key
- **WHEN** an input artifact's content changes (e.g. different Corpus text)
- **THEN** `cache_key` returns a different string

### Requirement: Artifacts round-trip through save and load
The system SHALL provide `save(key, artifacts, cache_dir)` and `load(key, cache_dir)` functions. Saving an artifact dict then loading it with the same key SHALL return a dict whose values are equivalent to the originals. If no artifact exists for a key, `load` SHALL return `None`.

#### Scenario: Corpus round-trips
- **WHEN** a `Corpus` artifact is saved then loaded with the same key
- **THEN** the loaded artifact has the same `text` field

#### Scenario: Dataset round-trips
- **WHEN** a `Dataset` artifact is saved then loaded with the same key
- **THEN** the loaded tensor equals the original tensor element-wise

#### Scenario: Policy round-trips
- **WHEN** a `Policy` artifact is saved then loaded with the same key
- **THEN** the loaded model's state dict equals the original model's state dict

#### Scenario: CharTokenizer round-trips
- **WHEN** a `CharTokenizer` artifact is saved then loaded with the same key
- **THEN** the loaded tokenizer has the same `char_to_id` mapping

#### Scenario: Cache miss returns None
- **WHEN** `load` is called with a key that has never been saved
- **THEN** it returns `None`

### Requirement: Warn when working tree is dirty
At runner start-up, if the git working tree has uncommitted changes, the runner SHALL emit a `UserWarning` naming the cache directory. This warns developers that cached results may not reflect their latest edits. When git is unavailable or the directory is not a git repository, no warning is emitted.

#### Scenario: Dirty worktree emits UserWarning naming cache dir
- **WHEN** the runner is called and the working tree has uncommitted changes
- **THEN** a `UserWarning` is emitted that includes the cache directory path

#### Scenario: Clean worktree emits no warning
- **WHEN** the runner is called and the working tree is clean
- **THEN** no `UserWarning` is emitted

### Requirement: Git ID falls back gracefully when git is unavailable
The system SHALL attempt to read the git short commit hash at module import time. If git is not installed or the working directory is not a git repository, it SHALL fall back to an empty string without raising an error.

#### Scenario: No git binary installed
- **WHEN** the `git` executable is not found on PATH
- **THEN** the module imports without error and uses `""` as the git ID component

#### Scenario: Not a git repository
- **WHEN** the working directory is not inside a git repository
- **THEN** the module imports without error and uses `""` as the git ID component
