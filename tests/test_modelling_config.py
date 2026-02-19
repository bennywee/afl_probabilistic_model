"""Tests for src.modelling.config module."""

import yaml
from pathlib import Path


class TestConfigLoading:
    """Tests for configuration loading from YAML."""

    def test_config_loads_data_config(self):
        """Test that DATA_CONFIG is loaded from YAML."""
        from src.modelling.config import DATA_CONFIG

        assert isinstance(DATA_CONFIG, dict)
        assert "ladder_path" in DATA_CONFIG
        assert "results_path" in DATA_CONFIG
        assert "fixture_path" in DATA_CONFIG
        assert "predict_round" in DATA_CONFIG

    def test_config_predict_round_is_tuple(self):
        """Test that predict_round is converted to a tuple."""
        from src.modelling.config import DATA_CONFIG

        assert isinstance(DATA_CONFIG["predict_round"], tuple)
        assert len(DATA_CONFIG["predict_round"]) == 2
        assert isinstance(DATA_CONFIG["predict_round"][0], int)  # Season
        assert isinstance(DATA_CONFIG["predict_round"][1], int)  # Round

    def test_config_paths_are_strings(self):
        """Test that path values are strings."""
        from src.modelling.config import DATA_CONFIG

        assert isinstance(DATA_CONFIG["ladder_path"], str)
        assert isinstance(DATA_CONFIG["results_path"], str)
        assert isinstance(DATA_CONFIG["fixture_path"], str)

class TestConfigDefaults:
    """Tests for configuration defaults."""

    def test_config_has_sensible_defaults(self):
        """Test that configuration has sensible default values."""
        from src.modelling.config import DATA_CONFIG

        # Ladder path should reference dev data
        assert "data" in DATA_CONFIG["ladder_path"]
        assert "ladder" in DATA_CONFIG["ladder_path"]

        # Results path should reference dev data
        assert "data" in DATA_CONFIG["results_path"]
        assert "results" in DATA_CONFIG["results_path"]

        # Fixture path should reference dev data
        assert "data" in DATA_CONFIG["fixture_path"]
        assert "fixture" in DATA_CONFIG["fixture_path"]

        # Predict round should be reasonable
        season, round_num = DATA_CONFIG["predict_round"]
        assert season > 1900  # Should be a real year
        assert 0 < round_num < 50  # Should be a valid round

    def test_config_yaml_file_exists(self):
        """Test that the config YAML file exists."""
        config_path = Path("configs") / "data_config.yaml"
        assert config_path.exists(), f"Config file not found at {config_path}"

    def test_config_yaml_valid_format(self):
        """Test that config YAML file is valid."""
        config_path = Path("configs") / "data_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        assert isinstance(config, dict)
        assert "modelling" in config
