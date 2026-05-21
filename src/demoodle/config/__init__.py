"""Load and expose the demoodle configuration."""

from pathlib import Path

import platformdirs
from configaroo import Configuration

from demoodle.config.schemas import DemoodleConfig

_config_path = Path(__file__).parent / "demoodle.toml"

config: DemoodleConfig = (
    Configuration.from_file(_config_path)
    .add_envs({"ARCHITECTURE": "architecture.active"}, prefix="DEMOODLE_")
    .parse_dynamic({"cache_path": platformdirs.user_cache_dir("demoodle")})
    # project_path set to project root by configaroo
    .convert_model(DemoodleConfig)
)
