# Spec: cli-train

## Purpose

The `demoo train` command provides a CLI entry point for the full day-one training pipeline. It assembles corpus → tokenizer → dataset → pretrain stages via the stage runner, and displays live training progress using a braille sparkline. Subsequent runs with unchanged config and code hit the artifact cache and complete immediately.

## Requirements

### Requirement: `demoo train` runs the full day-one pipeline and displays live training progress
The system SHALL provide a `demoo train` command that assembles the corpus → tokenizer → dataset → pretrain pipeline, executes it via the stage runner, and displays a live braille sparkline with current step and loss during training. On subsequent runs where all stages are cached, the command SHALL complete immediately with a cache-hit message.

#### Scenario: First run trains and shows sparkline
- **WHEN** `demoo train` is invoked and no cached model exists
- **THEN** the pipeline runs, a live two-line display (step/loss header + braille sparkline) updates throughout training, and the command exits 0 when training completes

#### Scenario: Cached run skips training
- **WHEN** `demoo train` is invoked a second time with no config or code changes
- **THEN** the command reports that all stages hit cache and exits without re-training

#### Scenario: Loss decreases over training
- **WHEN** `demoo train` completes on the names corpus
- **THEN** the final loss recorded in `TrainingMetrics` is lower than the initial loss

### Requirement: The sparkline fills left-to-right and doubles as a progress indicator
The system SHALL display the braille sparkline as a fixed-width strip pre-filled with empty braille cells (`⠀`). As training progresses, cells are filled left-to-right with height proportional to the loss at that window. The boundary between filled and empty cells indicates current progress; the strip is completely filled when training finishes.

#### Scenario: Sparkline encodes loss height
- **WHEN** loss is high early in training
- **THEN** the leftmost filled cells use tall braille glyphs (e.g. `⣿` or `⣶`)

#### Scenario: Sparkline encodes progress
- **WHEN** training is at 50% of n_steps
- **THEN** approximately half the cells in the strip are filled and half are empty (`⠀`)
