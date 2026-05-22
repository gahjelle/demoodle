## MODIFIED Requirements

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
