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
    "ladder_path": _mod_cfg.get("ladder_path"),
    "results_path": _mod_cfg.get("results_path"),
    "fixture_path": _mod_cfg.get("fixture_path"),
    "predict_round": (_mod_cfg.get("predict_round").get("season"), _mod_cfg.get("predict_round").get("round"))
}

# Feature names used in the model
FEATURE_NAMES = [
    _mod_cfg.get("feature_names")
]
