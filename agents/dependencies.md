# Dependencies

We use the following libraries (in alphabetical order):

- `configaroo`: Configuration loading from TOML and environment variables
- `cyclopts`: CLI handling
- `pydantic`: Configuration schema validation and typed config models
- `pytorch`: Training and running of models. PyTorch emits a `UserWarning` ("Failed to initialize NumPy") when its optional NumPy bridge is absent — this is suppressed at package level in `src/demoodle/__init__.py` since NumPy is not a dependency.
- `rich`: Console formatting
- `textual`: Terminal based UI
