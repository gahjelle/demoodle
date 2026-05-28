## ADDED Requirements

### Requirement: _generate pads short sequences to context_length
The system SHALL ensure that when `_generate` builds the context window for `arch.call`, the sequence is always exactly `context_length` tokens long. When `token_ids` has fewer than `context_length` tokens, it SHALL prepend the `\n` token (via `tokenizer.encode("\n")[0]`) to reach exactly `context_length`. The architecture SHALL always receive a full-length context; cold-start handling is the caller's responsibility.

#### Scenario: Short prompt is padded to context_length before arch.call
- **WHEN** `_generate` is called with a single-character prompt and `context_length=2`
- **THEN** the sequence passed to `arch.call` has length 2, with `\n` prepended

#### Scenario: Prompt at or above context_length is not padded
- **WHEN** `_generate` is called with a prompt of length >= `context_length`
- **THEN** the sequence passed to `arch.call` is `token_ids[-context_length:]` with no prepended tokens

#### Scenario: Bigram (context_length=1) is unaffected
- **WHEN** `_generate` is called with the bigram architecture and any single-character prompt
- **THEN** behaviour is identical to before (prompt is always >= context_length=1, no padding occurs)
