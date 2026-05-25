# Trainadillo — Learning Docs

Trainadillo is a minimal PyTorch clone built on NumPy. Its purpose is to
understand how PyTorch works under the hood by reimplementing its core concepts
from scratch — computation graphs, autograd, modules, optimizers — without the
GPU kernels, C++ tape, or production complexity.

These docs are the **learning companion to the implementation**. Each file
corresponds to one work item from `PLANS_TRAINADILLO.md` and answers three
questions:

1. **What concept does this introduce?** The idea, not the code.
2. **Why is it designed this way?** The invariants, the tradeoffs, the
   non-obvious decisions.
3. **How does real PyTorch do it?** Where trainadillo is faithful, where it
   simplifies, and what the C++/CUDA layer adds that we omit.

## File naming

Each doc maps to its T-item:

```
T1-tensor-class.md
T2-creation-functions.md
T3-rng.md
...
```

## What to include

- **Concept**: what problem this solves in PyTorch's architecture
- **Key invariants**: things that must stay true for later pieces to work
  (e.g. "integer indexing always returns a Tensor, never a scalar")
- **Design decisions**: choices made during implementation and the reasoning
- **PyTorch comparison**: where real PyTorch goes further — and why that
  extra complexity exists
- **What comes next**: which future T-items build on this concept

These docs should read like the notes you'd take while studying PyTorch's source
— not API documentation, but explanation of *why the thing is shaped the way it is*.
