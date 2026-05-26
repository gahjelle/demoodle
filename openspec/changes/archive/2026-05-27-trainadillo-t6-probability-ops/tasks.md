## 1. Refactor `_resolve_generator` to `_rng.py`

- [x] 1.1 Move `_resolve_generator` from `src/trainadillo/_random.py` to `src/trainadillo/_rng.py`, placing it after the `Generator` class definition
- [x] 1.2 Update `src/trainadillo/_random.py` to import `_resolve_generator` from `trainadillo._rng` instead of defining it

## 2. Implement `softmax`, `cumsum`, `multinomial` in `_ops.py`

- [x] 2.1 Add `softmax(tensor: Tensor, dim: int = -1) -> Tensor` using numerically stable log-sum-exp (`keepdims=True` on max and sum)
- [x] 2.2 Add `cumsum(tensor: Tensor, dim: int) -> Tensor` delegating to `np.cumsum(data, axis=dim)`
- [x] 2.3 Add `multinomial(probs: Tensor, num_samples: int, *, generator: Generator | None = None) -> Tensor` using `_resolve_generator` and `rng._np_rng.choice` with float64-normalized probs and `replace=False`

## 3. Tests in `test_ops.py`

- [x] 3.1 Test `softmax`: output sums to 1.0, no NaN/Inf on extreme inputs (`1000.0`, `-1000.0`), correct shape preserved
- [x] 3.2 Test `softmax` with `dim=0` on a 2-D tensor: each column sums to 1.0
- [x] 3.3 Test `cumsum`: result equals `np.cumsum` on the underlying data; output shape equals input shape
- [x] 3.4 Test `multinomial` output: shape is `(num_samples,)`, dtype is int64
- [x] 3.5 Test `multinomial` reproducibility: same seed → same index; different seeds → different indices
- [x] 3.6 Test `multinomial` distribution: 10 000 draws from a known distribution, assert each index frequency is within `0.02` of its true probability

## 4. Educational doc

- [x] 4.1 Write `docs/trainadillo/006-probability-ops.md` covering: why numerically stable softmax matters (overflow/underflow), what cumsum does in the nucleus-sampling pipeline, how multinomial maps probs to indices via numpy's `choice`, and the float32→float64 normalization step

## 5. Verification and housekeeping

- [x] 5.1 Run `uv run ruff format src/trainadillo/ tests/trainadillo/`
- [x] 5.2 Run `uv run ruff check src/trainadillo/ tests/trainadillo/`
- [x] 5.3 Run `uv run ty check src/trainadillo/ tests/trainadillo/`
- [x] 5.4 Run `uv run pytest tests/trainadillo/`
- [x] 5.5 Mark T6 as ✅ in `PLANS_TRAINADILLO.md`
- [x] 5.6 Review `CONTEXT.md` and add any new terms if needed
