## Why

The bigram architecture demonstrates single-token prediction but cannot capture two-character patterns (e.g. "th" → "e", "br" → common vowel). Adding a trigram model provides a natural next step on the architecture axis — showing how extending context from 1 to 2 tokens improves character-level predictions, without yet introducing the complexity of embeddings or hidden layers.

## What Changes

- **New**: `architectures/sampling.py` — extracts `_sample` from `bigram.py` and renames it to `sample` (public), so all architectures share one sampling implementation
- **New**: `architectures/trigram.py` — `TrigramModel` (V×V×V weight tensor) and `TrigramArchitecture` (context_length=2)
- **Modified**: `architectures/bigram.py` — imports `sample` from the shared module instead of defining it locally
- **Modified**: `frontends/cli.py` — `_generate` pads short sequences with the `\n` token when context is shorter than `context_length`; `_make_arch` gains a `"trigram"` case
- **Modified**: `config/schemas.py` — adds `TrigramConfig` and wires it into `ArchitecturesConfig`
- **Modified**: `config/demoodle.toml` — adds `[architecture.trigram]` section
- **Modified**: architecture and context documentation to reflect trigram's existence

## Capabilities

### New Capabilities

- `trigram-architecture`: A V×V×V weight tensor model predicting from the last two tokens. Satisfies `ArchitectureProtocol` and `InspectableProtocol`. Selectable via `architecture.active = "trigram"` in config.
- `shared-sampling`: Public `sample` function in `architectures/sampling.py` shared by all architecture implementations.

### Modified Capabilities

- `bigram-architecture`: No requirement changes. Implementation refactored to import `sample` from the shared module.
- `cli-call`: `_generate` now pads sequences shorter than `context_length` with the `\n` token, ensuring architectures always receive exactly `context_length` tokens.

## Non-goals

- No PLANS.md work item — this is an informal side branch for exploration and comparison
- No automatic architecture switching — users change `active` in the config file manually
- No support for N-gram generalisation beyond trigram (4-gram etc.)
- No performance optimisation for large vocabularies (V³ scaling is documented as a known limitation)

## Impact

- `tests/architectures/test_bigram.py` — import of `_sample` updated to `sample` from new location
- `tests/architectures/test_trigram.py` — new test file
- `docs/architectures/trigram.md` — new educational doc
- `docs/architectures/bigram.md` — relations section updated
- `CONTEXT.md` — `ArchitectureProtocol` `context_length` description updated
- `src/demoodle/architectures/__init__.py` — docstring updated
