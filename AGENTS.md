# Demoodle - Demonstrate how (Large) Language Models are built

This is a Python 3.14+ project that uses `uv` for package management. Always use
`uv run ...` instead of `python3 ...` to run any Python code. Use `uv add ...`
when adding dependencies. Don't edit `pyproject.toml` manually.

## Development Commands

Use Pytest, Ruff, Ty for testing and linting:

```bash
# Install dependencies
uv sync

# Run the TUI (main chat interface)
uv run demoodle

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/path/to/test_file.py::test_name

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run ty check .
```

## General Code Principles

- Functional Core/Imperative Shell architecture
- Always prefer immutable operations over mutable. Use `list` not `tuple` for homogeneous sequences, but prefer immutable operations.
- Comments explain why, not what. Do not add signature information to docstrings
- Ruff for linting and formatting with all rules enabled only COM812, D203, D213 are disabled. Use inline overrides only when truly necessary
- Type checker is `ty` (not `mypy`), use `# ty: ignore[<code>]` if suppression is ever needed, never `# type: ignore[<mypy-code>]`

## More Information

You can find more information about the project inside agents/

Read and write files inside agents/ as necessary. You can find an index of these
files in agents/README.md. Never add more information to this `AGENTS.md` file.
