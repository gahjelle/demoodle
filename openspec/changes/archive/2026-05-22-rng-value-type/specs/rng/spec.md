## ADDED Requirements

### Requirement: RNG is an immutable value type
`RNG` SHALL be a frozen dataclass holding a single integer seed. It MUST NOT carry any mutable state or reference any global random state.

#### Scenario: RNG is immutable
- **WHEN** code attempts to assign to a field on an `RNG` instance
- **THEN** a `FrozenInstanceError` is raised

#### Scenario: RNG holds its seed
- **WHEN** `RNG(seed=42)` is constructed
- **THEN** `rng.seed == 42`

### Requirement: split produces two deterministic, diverging children
`RNG.split()` SHALL return a tuple of exactly two new `RNG` instances. Given the same parent seed, repeated calls to `.split()` MUST return identical pairs. The two children MUST have different seeds from each other.

#### Scenario: split is deterministic
- **WHEN** `rng.split()` is called twice on the same instance
- **THEN** both calls return identical `(left, right)` pairs

#### Scenario: split children diverge
- **WHEN** `left, right = rng.split()`
- **THEN** `left.seed != right.seed`

#### Scenario: different parents produce different children
- **WHEN** two RNGs with different seeds are each split
- **THEN** their left children have different seeds, and their right children have different seeds

### Requirement: generator returns a seeded torch.Generator
`RNG.generator()` SHALL return a `torch.Generator` seeded deterministically from `self.seed`. Calling `.generator()` multiple times on the same `RNG` instance MUST return generators that produce identical random sequences.

#### Scenario: generator is reproducible
- **WHEN** `rng.generator()` is called twice
- **THEN** both generators produce the same sequence of draws

#### Scenario: different seeds produce different draws
- **WHEN** two RNGs with different seeds each call `.generator()`
- **THEN** the resulting generators produce different random sequences
