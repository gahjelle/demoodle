## Why

The project needs a typed, validated configuration layer before any real functionality can be built. Without it, every component that needs a hyperparameter (architecture type, learning rate, context length, etc.) would have to reach for ad-hoc values or untyped dicts, making the system hard to demo and impossible to drive from a single config file or environment variable.

## What Changes

- Add `pydantic` as a project dependency
- Add `src/demoodle/config/` package with three files:
  - `schemas.py` — nested pydantic `BaseModel` classes defining all configuration sections
  - `demoodle.toml` — default configuration in TOML format, organised by axis
  - `__init__.py` — loads and exposes a typed `config` singleton via `configaroo`
- Remove the planned core `Config` frozen dataclass (W2) — `DemoodleConfig` and its section models replace it throughout
- `DEMOODLE_ARCHITECTURE` environment variable overrides the active architecture at runtime

## Capabilities

### New Capabilities

- `app-config`: Typed, validated application configuration loaded from TOML and overridable via environment variables. Exposes a `config` singleton importable as `from demoodle.config import config`.

### Modified Capabilities

_(none — no existing specs)_

## Impact

- **New dependency**: `pydantic`
- **New package**: `src/demoodle/config/`
- **W2 design change**: core `Config` frozen dataclass is dropped; pydantic section models (`MLPConfig`, `TransformerConfig`, etc.) are passed directly into `init_state()` and stage functions instead
- **No existing code broken**: the project is at skeleton stage; only `core/__init__.py` references the planned `Config` type, which has not been implemented yet

## Non-goals

- Not a CLI argument parser — cyclopts handles CLI; this is file/env configuration only
- Not dynamic at runtime — config is loaded once at startup; live reloading is out of scope
- Not all hyperparameters for all future architectures — BPE and transformer sections will be stubs with placeholder defaults until those work items are built
