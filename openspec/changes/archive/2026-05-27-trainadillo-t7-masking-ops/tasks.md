## 1. Demoodle update (validate against real PyTorch first)

- [x] 1.1 In `src/demoodle/architectures/bigram.py`, rewrite the top-k `scatter_` call site: replace the 3-line `mask = ...; mask.scatter_(...); scaled = mask` block with a single `scaled = torch.full_like(...).scatter(0, top_indices, scaled[top_indices])`
- [x] 1.2 In `src/demoodle/architectures/bigram.py`, rewrite the top-p `scatter_` call site: replace `torch.zeros_like(scaled).scatter_(0, ...)` with `torch.zeros_like(scaled).scatter(0, ...)`
- [x] 1.3 Run `uv run pytest tests/architectures/test_bigram.py` — must pass against real PyTorch, confirming the non-in-place rewrite is correct

## 2. Implement `masked_fill` and `scatter` on Tensor

- [x] 2.1 Add `Tensor.masked_fill(self, mask: Tensor, value: float) -> Tensor` to `src/trainadillo/_tensor.py`: copy `self._data`, set positions where `mask._data` is True to `value`, return new Tensor
- [x] 2.2 Add `Tensor.scatter(self, dim: int, index: Tensor, src: Tensor) -> Tensor` to `src/trainadillo/_tensor.py`: copy `self._data`, apply `out[index._data] = src._data` (fancy indexing for dim=0 1-D case), return new Tensor

## 3. Unit tests

- [x] 3.1 Test `masked_fill`: True positions contain fill value; False positions are unchanged; `self` is not mutated; works with `float("-inf")`
- [x] 3.2 Test `scatter` basic: values placed at correct indices; unindexed positions unchanged; `self` not mutated
- [x] 3.3 Test `scatter` sort-inversion: `zeros_like(x).scatter(0, sorted_indices, sorted_values)` recovers the original unsorted tensor

## 4. PyTorch compatibility tests (temporary — delete at T20)

- [x] 4.1 Create `tests/trainadillo/test_compat_pytorch.py` with `torch = pytest.importorskip("torch")` guard at module level
- [x] 4.2 Test exact match for deterministic sub-ops: `softmax`, `scatter`, `masked_fill`, `cumsum` — run identical inputs through both libraries, assert `np.allclose(rtol=1e-5)`
- [x] 4.3 Implement a `_probs(lib, logits, temperature, top_k, top_p)` helper that replicates the `_sample()` filtering logic (everything up to but not including `multinomial`) parameterised by library
- [x] 4.4 Test pipeline probs match: for `(top_k=2, top_p=None)`, `(top_k=None, top_p=0.9)`, `(top_k=None, top_p=None)` — assert `np.allclose` on the probs tensors from both libraries
- [x] 4.5 Test corner case `top_k=1`: both libraries return the argmax token (exact agreement, no randomness)
- [x] 4.6 Test corner case `top_p=0.0`: both libraries return the argmax token (exact agreement, no randomness)

## 5. Verification and housekeeping

- [x] 5.1 Run `uv run ruff format src/trainadillo/ tests/trainadillo/ src/demoodle/architectures/`
- [x] 5.2 Run `uv run ruff check src/trainadillo/ tests/trainadillo/ src/demoodle/architectures/`
- [x] 5.3 Run `uv run ty check src/trainadillo/ tests/trainadillo/ src/demoodle/architectures/`
- [x] 5.4 Run `uv run pytest` — full suite must pass
- [x] 5.5 Write `docs/trainadillo/007-masking-ops.md` covering: what masked_fill does and why `-inf` is the right fill for logit filtering; what scatter does and the conceptual difference between scatter (write to positions) and gather (read from positions); why the non-in-place form is preferred; the temporary compat test strategy
- [x] 5.6 Mark T7 as ✅ in `PLANS_TRAINADILLO.md`
- [x] 5.7 Review `CONTEXT.md` and add any new terms if needed
