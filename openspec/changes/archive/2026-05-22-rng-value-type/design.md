## Context

Pipeline stages need random number generation for model initialization, data shuffling, and token sampling. PyTorch's default global seed is fragile — inserting any random operation anywhere shifts all subsequent draws. The `RNG` type makes randomness explicit: each stage receives an `RNG` value, uses it locally, and the two subsystems cannot interfere with each other.

`W2` (core types) is complete. `W3` adds a single new module that depends on nothing except Python's stdlib and PyTorch (both already present).

## Goals / Non-Goals

**Goals:**
- Frozen, immutable `RNG` dataclass in `core/rng.py`
- `.split()` producing two deterministic, diverging children using `hashlib.sha256`
- `.generator()` returning a seeded `torch.Generator`
- Property tests covering determinism, divergence, and draw reproducibility

**Non-Goals:**
- Global seeding helpers
- Integration with numpy or stdlib `random`
- Inclusion in the `Artifact` union (RNG is threading infrastructure, not a pipeline output)
- RNG serialization or checkpointing

## Decisions

### Mixing function: `hashlib.sha256` over arithmetic constants or `hash()`

`hash()` is randomized by `PYTHONHASHSEED` across Python processes — unusable for reproducibility. LCG arithmetic constants (e.g., Knuth multipliers) are equally stable but require domain knowledge to read and verify. `hashlib.sha256` is immediately legible: a non-specialist can confirm it produces a stable, well-distributed digest without knowing anything about PRNGs.

```python
import hashlib

def _mix(seed: int, tag: int) -> int:
    digest = hashlib.sha256(f"{seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "little")

# split() produces:
left  = RNG(seed=_mix(self.seed, 0))
right = RNG(seed=_mix(self.seed, 1))
```

The two tags (`0` and `1`) guarantee the children are independent of each other. The parent seed is mixed in, so different parents produce different children.

### Return shape: fixed pair `(RNG, RNG)`

Mirrors JAX's `jax.random.split`, which ML practitioners recognize. A three-way split chains two calls explicitly, making the tree structure visible in the code rather than hiding it behind a `split(n=3)` overload.

### Seed type: `int` (unbounded)

Python `int` is unbounded; we mask to 64 bits when constructing a `torch.Generator` (`generator.manual_seed(self.seed & 0xFFFF_FFFF_FFFF_FFFF)`). This avoids overflow and matches PyTorch's accepted seed range.

### Location: `core/rng.py`, not `core/types.py`

`types.py` holds `Artifact` variants — values that flow between stages and get cached. `RNG` is threading infrastructure passed *into* stages alongside artifacts. Separating them keeps the import surface clean and signals the different role.

## Risks / Trade-offs

- **Seed collisions**: SHA-256 over a 64-bit space has negligible collision probability — not a practical risk.
- **Teaching overhead**: Developers unfamiliar with functional RNG threading may find the split pattern unfamiliar. Mitigated by the explicit API and the educational context of this project.
- **Tag exhaustion**: Using only tags `0` and `1` means `.split()` always produces exactly two children. If a future stage needs N > 2 independent streams, it chains calls. This is by design (see pair decision above) — the awkwardness of chaining is intentional signal.

## Open Questions

None. All decisions are resolved above.
