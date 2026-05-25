# W1 — Project Skeleton & Tooling

## The `src/` layout

Most Python tutorials put source files directly in the project root:
`demoodle/foo.py`. The `src/` layout moves them one level deeper:
`src/demoodle/foo.py`. The difference is subtle but important for testing.

Without `src/`, running `pytest` from the project root can silently import the
development tree instead of the installed package. If a file exists locally but
is not declared in `pyproject.toml`, tests pass locally and fail for users. The
`src/` layout forces a proper install (`uv sync` / `pip install -e .`) and breaks
the accidental import path.

This is more than hygiene in ML: models are distributed as packages, and the test
suite often validates behavior of serialized artifacts that were built by the
installed package, not raw source files.

## `uv` as a package manager

`uv` replaces pip + virtualenv + pip-tools with a single fast tool. Its lockfile
(`uv.lock`) pins every transitive dependency to an exact version and hash.

For ML work, lockfiles are not optional. Upgrading `numpy` from 1.26 to 2.0, or
`torch` from 2.3 to 2.5, can silently change model behavior — different
floating-point defaults, different random initialization, different operator
behavior. Reproducible environments are the prerequisite for reproducible results.

## `ruff`, `pytest`, `ty`

Ruff is a linter and formatter (replacing flake8 + black + isort). `ty` is a fast
type checker (replacing mypy). `pytest` is the test runner.

The practical ML reason to run these continuously: type errors in model code often
surface as runtime crashes deep inside a training loop, far from their cause. A
type checker that catches `Tensor` vs `int` confusion at edit time saves hours of
debugging.
