"""Plain CLI entry point for the demoo command."""

import configaroo
import cyclopts

from demoodle.config import config

app = cyclopts.App(name="demoo", help="Demoo - build your own (large) language models")


@app.command
def show_config(section: str | None = None) -> None:
    """Show the Demoodle configuration."""
    configaroo.print_configuration(config, section=section)
