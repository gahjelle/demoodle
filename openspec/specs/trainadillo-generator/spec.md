# Spec: Trainadillo Generator

## Purpose

The `trainadillo-generator` capability provides a controllable random number
generator with PyTorch's `Generator` interface. It wraps numpy's PCG64 generator
and enforces explicit seeding at the module level, making non-reproducible random
ops a loud error rather than a silent one.

## Requirements

### Requirement: `Generator()` is immediately usable

`Generator()` with no arguments SHALL be seeded from OS entropy and SHALL be in a
valid, usable state immediately after construction without calling `manual_seed()`.

#### Scenario: Generator is usable immediately after construction
- **GIVEN** `g = Generator()`
- **THEN** `g._np_rng` is a `numpy.random.Generator` instance
- **AND** no error is raised on construction

### Requirement: `manual_seed` re-seeds and returns `self`

`Generator.manual_seed(seed)` SHALL re-initialize the internal PCG64 with the given
integer seed, set `_seeded = True`, and return `self`.

#### Scenario: Same seed reproduces the same sequence
- **GIVEN** `g = Generator()` and `g.manual_seed(42)`
- **WHEN** draws are taken, then `g.manual_seed(42)` is called again
- **THEN** the second sequence of draws is identical to the first

#### Scenario: Different seeds diverge
- **GIVEN** two Generators each seeded with different integers
- **WHEN** both draw the same number of values from `_np_rng`
- **THEN** the sequences differ

#### Scenario: `manual_seed` returns `self`
- **GIVEN** `g = Generator()`
- **WHEN** `result = g.manual_seed(42)`
- **THEN** `result is g`

### Requirement: `repr` reflects seeded state

`repr(generator)` SHALL return `"Generator(unseeded)"` for OS-entropy generators
(no `manual_seed` called) and `"Generator(seeded)"` after `manual_seed()` is called.

#### Scenario: repr before seeding
- **GIVEN** `g = Generator()`
- **THEN** `repr(g) == "Generator(unseeded)"`

#### Scenario: repr after seeding
- **GIVEN** `g = Generator()` and `g.manual_seed(42)` called
- **THEN** `repr(g) == "Generator(seeded)"`

### Requirement: Module-level `manual_seed` seeds the default generator and returns it

`trainadillo.manual_seed(seed)` SHALL set the module-level `_default_generator` to a
`Generator` seeded with `seed` (or re-seed it if already set) and return it.

#### Scenario: manual_seed returns a seeded Generator
- **GIVEN** `g = trainadillo.manual_seed(42)`
- **THEN** `repr(g) == "Generator(seeded)"`
- **AND** `g` is the module-level `_default_generator`

#### Scenario: Two calls with the same seed produce identical subsequent sequences
- **GIVEN** `trainadillo.manual_seed(42)` is called, draws taken, then
  `trainadillo.manual_seed(42)` called again
- **THEN** the second sequence of draws matches the first

### Requirement: Module-level default is `None` until `manual_seed` is called

`_default_generator` SHALL be `None` at module load. T4's random functions SHALL
check for `None` and raise `RuntimeError` if they encounter it.

#### Scenario: Default is None before manual_seed
- **GIVEN** a fresh module state
- **THEN** `trainadillo._rng._default_generator is None`

#### Scenario: Default is set after manual_seed
- **GIVEN** `trainadillo.manual_seed(42)` has been called
- **THEN** `trainadillo._rng._default_generator is not None`
