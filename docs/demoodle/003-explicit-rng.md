# W3 — RNG Value Type

## The problem with global random state

Most ML code seeds randomness with `torch.manual_seed(42)` at startup and then
forgets about it. This works for simple scripts, but breaks when:

- **Stages run in a different order**: changing the graph topology changes which
  random numbers each stage draws, producing different results from the same seed
- **Caching skips a stage**: the downstream stage now draws from a different RNG
  position than it did during the cached run, breaking reproducibility
- **Parallelism is introduced**: different processes inherit copies of the same
  global state and produce identical "random" sequences

The root cause is that global random state is **implicit** — the sequence of
random numbers a function draws depends on everything that happened before it in
program execution, which is not part of its interface.

## JAX's explicit keys

JAX solved this in 2018 with a functional RNG design: every function that needs
randomness receives an explicit key, and new keys are created by splitting:

```python
key, subkey = jax.random.split(key)
samples = jax.random.normal(subkey, shape=(100,))
```

Splitting is deterministic: the same `key` always produces the same pair of
children. Every function's randomness is now a pure function of its explicit
inputs, independent of call order or program history.

Demoodle's `RNG` takes the same stance. Each stage receives a child RNG via
`.split()`, derived from the root seed:

```python
stage_rng, rng = rng.split()
outputs = stage.run(inputs, stage_rng)
```

Stages can be reordered, cached, or parallelized without changing each stage's
random draws — as long as their position in the topological sort is stable.

## Why SHA-256 and not a simpler approach

A naive split might produce `RNG(seed + 1)` and `RNG(seed + 2)`. The problem:
if you split twice from adjacent seeds, you can get overlapping random sequences.
Demoodle uses SHA-256 to mix the seed and a tag (`0` or `1`) into a 256-bit
digest:

```python
def _mix(seed: int, tag: int) -> int:
    digest = hashlib.sha256(f"{seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "little")
```

The hash makes splitting order-independent and statistically sound: outputs are
pseudorandom regardless of how close the input seeds are. NumPy uses a similar
approach in its `SeedSequence` abstraction, which hashes integer seeds through
a mixing function before initializing the underlying PRNG.

## `generator()` as the bridge to PyTorch

PyTorch's tensor ops (`torch.randint`, `nn.init.normal_`) accept an optional
`generator` argument — a local, stateful `torch.Generator`. `rng.generator()`
creates one seeded from `rng.seed`:

```python
def generator(self) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(self.seed & 0xFFFF_FFFF_FFFF_FFFF)
    return g
```

This lets Demoodle's explicit RNG model interoperate with PyTorch's existing API
without changing PyTorch. The generator is created fresh each call — it is not
stored on `RNG`, which is a frozen dataclass with only a `seed` field.
