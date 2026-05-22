"""Load bundled corpora into Corpus artifacts."""

import importlib.resources

from demoodle.core.types import Corpus


def load_corpus(name: str) -> Corpus:
    """Load a named corpus from bundled package data."""
    data_pkg = importlib.resources.files("demoodle.data")
    resource = data_pkg / f"{name}.txt"
    try:
        text = resource.read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError) as exc:
        msg = f"No bundled corpus named {name!r}"
        raise FileNotFoundError(msg) from exc
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return Corpus(text="\n".join(lines))
