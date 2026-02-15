"""Configuration and constants for the modeling pipeline.

This module loads runtime configuration from `configs/data_config.yaml`.
The modelling-specific parameters are located under the `modelling` key.
"""

from pathlib import Path
import yaml

# Path to the shared YAML config file
_CONFIG_PATH = Path("configs") / "data_config.yaml"

try:
    with _CONFIG_PATH.open("r") as fh:
        _cfg = yaml.safe_load(fh) or {}
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("PyYAML is required to load configs/data_config.yaml. Install with `pip install pyyaml`") from e

# Extract modelling config with sensible defaults
_mod_cfg = _cfg.get("modelling", {})

DATA_CONFIG = {
    "ladder_path": _mod_cfg.get("ladder_path", "data/dev/raw/ladder"),
    "results_path": _mod_cfg.get("results_path", "data/dev/raw/results"),
    "fixture_path": _mod_cfg.get("fixture_path", "data/dev/raw/fixture"),
}

_pr = _mod_cfg.get("predict_round") or {}
if isinstance(_pr, dict):
    DATA_CONFIG["predict_round"] = (_pr.get("season", 2025), _pr.get("round", 20))
else:
    DATA_CONFIG["predict_round"] = tuple(_pr) if _pr else (2025, 20)

# Feature names used in the model
FEATURE_NAMES = [
    "home_team_prev_percentage",
    "away_team_prev_percentage",
    "prev_home_delta_wins",
    "prev_home_delta_loss",
]
