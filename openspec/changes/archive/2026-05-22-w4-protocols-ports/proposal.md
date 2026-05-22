## Why

The pipeline needs explicit swap points — named, typed contracts that let architectures, tokenizers, and stages be exchanged without touching runner or training code. W2 gave us the data spine; W4 gives us the behavioral seams those types flow through. Without these protocols, W6 (runner) and W8 (CharTokenizer) have nothing stable to depend on.

## What Changes

- Add `ports/protocols.py` with `TokenizerProtocol`, `ArchitectureProtocol`, `InspectableProtocol`, and the `Stage` frozen dataclass
- All protocol names carry the `Protocol` suffix to avoid shadowing the artifact types in `core/types` (e.g. `Tokenizer` artifact vs `TokenizerProtocol` behavioral contract)
- `Stage.run` includes `RNG` in its signature, enforcing that stages are pure functions of their inputs — no global seeding
- No `FrontendProtocol`: the real frontend seam is the W26 shell API; CLI/TUI/web are too different for a meaningful shared interface

## Non-goals

- Concrete implementations of any protocol (those are W8, W10, W12 onwards)
- Persistence, caching, or runner logic
- Frontend protocols or any presentation-layer contracts

## Capabilities

### New Capabilities

- `pipeline-ports`: The typed swap points for tokenizers, architectures, inspectable inference, and pipeline stages

### Modified Capabilities

*(none)*

## Impact

- New file: `src/demoodle/ports/protocols.py`
- `ports/__init__.py` gains exports for all four names
- Downstream work items (W6, W8, W10, W12) depend on these protocols being stable
- Depends on W2 (core types) and W3 (RNG value type)
