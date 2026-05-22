## Why

The pipeline has a corpus and (once W8 lands) a tokenizer, but no stage to connect
them into encoded training data. `build_dataset` is the bridge: it turns raw text
into a flat integer token sequence, which every architecture (bigram, MLP, transformer)
consumes directly.

## What Changes

- Add `src/demoodle/data/stages.py` with `make_build_dataset_stage()` factory
  - Stage `needs=["corpus", "tokenizer"]`, `produces=["dataset"]`
  - Encodes `corpus.text` (with trailing `\n` normalization) via `tokenizer.encode()`
  - Returns `Dataset(tokens=torch.tensor(..., dtype=torch.long))`
- Add `tests/data/test_stages.py` with shape, dtype, and shift-by-one tests

## Capabilities

### New Capabilities

- `dataset-builder`: Encode a corpus into a flat token-id tensor suitable for
  next-token prediction training; normalizes trailing `\n` so every name boundary
  appears as a training example

### Modified Capabilities

- `artifact-cache`: No changes — `_hash_artifact` already has a `Dataset` branch

## Impact

- New: `src/demoodle/data/stages.py`, `tests/data/test_stages.py`
- No changes to `core/types.py`, `shell/persistence.py`, or any existing stage
- Downstream: W10 (bigram) and W11 (pretrain) consume `artifacts["dataset"]`

## Non-goals

- Padding, truncation, or fixed-length chunking (W16 transformer concern)
- Batching logic (W11 training loop concern)
- Config-driven tokenizer selection (W19)
- Any changes to the `Dataset` frozen dataclass
