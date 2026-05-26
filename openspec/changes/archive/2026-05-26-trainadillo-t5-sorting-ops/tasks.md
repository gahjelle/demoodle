## 1. `Tensor.argmax` method

- [x] 1.1 Add `argmax(self, dim=None) -> Tensor` to `Tensor` in `src/trainadillo/_tensor.py`, delegating to `np.argmax(self._data, axis=dim)`

## 2. `_ops.py` — NamedTuples and sorting functions

- [x] 2.1 Create `src/trainadillo/_ops.py` with `TopKResult(values, indices)` NamedTuple
- [x] 2.2 Add `SortResult(values, indices)` NamedTuple to `_ops.py`
- [x] 2.3 Implement `topk(tensor, k, dim=-1) -> TopKResult` using `np.argsort`, `np.flip`, `np.take`, and `np.take_along_axis`
- [x] 2.4 Implement `sort(tensor, dim=-1, *, descending=False) -> SortResult` using `np.argsort` with `np.flip` for descending, and `np.take_along_axis`

## 3. Tests

- [x] 3.1 Create `tests/trainadillo/test_ops.py`
- [x] 3.2 Test `topk`: top-1, top-3, k=len, named-access, values in descending order
- [x] 3.3 Test `sort`: ascending, descending, index round-trip (`t[indices] == values`)
- [x] 3.4 Test `sort` rejects positional `descending` argument
- [x] 3.5 Test `argmax`: flat tensor returns 0-D Tensor, with `dim=` on a 2-D tensor, result is Tensor not int

## 4. Educational doc

- [x] 4.1 Write `docs/trainadillo/005-sorting-ops.md` covering: why these ops exist (sampling pipeline), argsort as permutation, take_along_axis as gather, descending-via-negation, why indices are returned alongside values, NamedTuples for ergonomic returns, argmax as greedy decoding

## 5. Verification and housekeeping

- [x] 5.1 Run `uv run ruff format src/trainadillo/_ops.py src/trainadillo/_tensor.py tests/trainadillo/test_ops.py`
- [x] 5.2 Run `uv run ruff check src/trainadillo/ tests/trainadillo/`
- [x] 5.3 Run `uv run ty check src/trainadillo/ tests/trainadillo/`
- [x] 5.4 Run `uv run pytest tests/trainadillo/`
- [x] 5.5 Mark T5 as ✅ in `PLANS_TRAINADILLO.md`
- [x] 5.6 Review `CONTEXT.md` and add any new terms if needed
