"""Configuration and constants for the modeling pipeline."""

# Data paths
DATA_CONFIG = {
    "ladder_path": "data/dev/raw/ladder",
    "results_path": "data/dev/raw/results",
    "fixture_path": "data/dev/raw/fixture",
    "predict_round": (2025, 20),
}

# Feature names used in the model
FEATURE_NAMES = [
    "home_team_prev_percentage",
    "away_team_prev_percentage",
    "prev_home_delta_wins",
    "prev_home_delta_loss",
]
