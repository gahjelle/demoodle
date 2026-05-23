# Spec: cli-call

## Purpose

The `demoo call` command provides a CLI entry point for generating text continuations from a trained policy. It runs the full pipeline (using the artifact cache when available), then generates `n` continuations via `arch.call()`. If no cached model is found, it auto-trains first with a notice to the user.

## Requirements

### Requirement: `demoo call` generates text continuations from the trained policy
The system SHALL provide a `demoo call` command that runs the full pipeline (using the cache when available), then generates `n` text continuations by repeatedly invoking `arch.call()` on the trained policy. Each continuation starts from `--prompt` (default `"\n"`) and stops on a newline token or after `max_len=100` tokens.

#### Scenario: Generates names after training
- **WHEN** `demoo call` is invoked after `demoo train` has run
- **THEN** the command prints `n` generated names (default 5), one per line, without re-training

#### Scenario: Temperature affects output distribution
- **WHEN** `demoo call --temperature 0.1` is invoked
- **THEN** the generated names are more peaked / repetitive compared to `--temperature 2.0`

#### Scenario: `--n` controls the number of generations
- **WHEN** `demoo call --n 10` is invoked
- **THEN** exactly 10 continuations are printed

#### Scenario: `--prompt` seeds the generation context
- **WHEN** `demoo call --prompt "Ma"` is invoked
- **THEN** each generated continuation begins with the characters `Ma`

### Requirement: `demoo call` auto-trains with a notice if no model is cached
The system SHALL detect when no cached `base_policy` exists and, instead of failing, print a single-line notice and run training (with the live sparkline display) before generating. No user intervention is required.

#### Scenario: Call before train prints notice and trains
- **WHEN** `demoo call` is invoked with an empty cache
- **THEN** a notice line is printed (e.g. "No trained model found — running train first."), training runs with the live display, and names are printed once training completes

#### Scenario: Call after train skips training
- **WHEN** `demoo call` is invoked after a prior `demoo train` run with unchanged config
- **THEN** no training display appears and names are printed immediately
