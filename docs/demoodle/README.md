# Demoodle — Learning Docs

Demoodle is a from-scratch implementation of the core components of an LLM system
— tokenizers, training pipelines, model architectures, and post-training — built
to make the *why* visible. These docs are the learning companion to the
implementation.

Each file corresponds to one work item from `PLANS.md` and answers three
questions:

1. **What concept does this introduce?** The idea, not the code.
2. **Why is it designed this way?** The invariants, the tradeoffs, the
   non-obvious decisions.
3. **How do real LLM systems do it?** Where Demoodle is faithful, where it
   simplifies, and what production systems add that we omit.

## File naming

Each doc maps to its W-item number:

```
001-project-skeleton.md
002-core-value-types.md
003-explicit-rng.md
...
```

## What to include

- **Concept**: what problem this solves in a real LLM system
- **Key invariants**: things that must stay true for later pieces to work
  (e.g. "every artifact is immutable so its hash is stable after creation")
- **Design decisions**: choices made during implementation and the reasoning
- **Prior art**: how real systems handle the same problem — HuggingFace, MLflow,
  JAX, Airflow, or whatever is most directly relevant — and where Demoodle
  simplifies
- **What comes next**: which future W-items build on this concept

These docs should read like the notes you'd take while studying how LLM systems
are built — not API documentation, but explanation of *why the thing is shaped
the way it is*.
