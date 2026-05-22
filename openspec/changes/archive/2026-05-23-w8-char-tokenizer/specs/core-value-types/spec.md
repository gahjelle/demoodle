## MODIFIED Requirements

### Requirement: Artifact union type
The module SHALL export `Artifact` as an open union of all pipeline values:
`Corpus | CharTokenizer | Dataset | Policy | Metrics`. `CharTokenizer` is imported
from `demoodle.tokenizers.char`. The union grows as concrete types are confirmed:
- **W18**: `BpeTokenizer` added
- **W21**: `RewardModel` and `PreferenceData` added

No shared base class or inheritance is required — types join the union because
stages produce and consume them.

#### Scenario: Each variant is a valid Artifact
- **WHEN** any of `Corpus`, `CharTokenizer`, `Dataset`, `Policy`, or `Metrics` is type-checked against `Artifact`
- **THEN** the type checker accepts it without error
