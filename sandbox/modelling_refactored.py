import polars as pl
import pymc as pm
import numpy as np
import arviz as az
from typing import Tuple, Dict, Any


# Configuration
DATA_CONFIG = {
    "ladder_path": "data/dev/raw/ladder",
    "results_path": "data/dev/raw/results",
    "fixture_path": "data/dev/raw/fixture",
    "predict_round": (2025, 29),
}

FEATURE_NAMES = [
    "home_team_prev_percentage",
    "away_team_prev_percentage",
    "prev_home_delta_wins",
    "prev_home_delta_loss",
]


def load_data(
    ladder_path: str, results_path: str, fixture_path: str
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load and perform initial transforms on parquet data."""
    ladder = pl.scan_parquet(ladder_path).collect()
    results = pl.scan_parquet(results_path).collect()
    fixture = (
        pl.scan_parquet(fixture_path)
        .collect()
        .select("Round", "Home.Team", "Away.Team")
        .with_columns(Round=pl.col("Round") - 1)
    )
    return ladder, results, fixture


def process_results(results: pl.DataFrame) -> pl.DataFrame:
    """Add home_win column and filter to regular season only."""
    return (
        results.with_columns(
            pl.when(pl.col("Home.Points") > pl.col("Away.Points"))
            .then(1)
            .otherwise(0)
            .alias("home_win")
        )
        .filter(~pl.col("Round").is_in(["QF", "EF", "SF", "PF", "GF"]))
        .select("Season", "Round.Number", "home_win", "Home.Team", "Away.Team")
    )


def calculate_win_loss_records(results_df: pl.DataFrame) -> pl.DataFrame:
    """Calculate cumulative wins/losses for each team."""
    home = results_df.drop("Away.Team").rename({"home_win": "win", "Home.Team": "Team"})
    away = (
        results_df.drop("Home.Team")
        .with_columns(win=1 - pl.col("home_win"))
        .drop("home_win")
        .rename({"Away.Team": "Team"})
        .select("Season", "Round.Number", "win", "Team")
    )

    return (
        pl.concat([home, away], how="vertical")
        .sort(["Season", "Round.Number"])
        .with_columns(
            loss=1 - pl.col("win"),
            total_wins=pl.col("win").cum_sum().over("Season", "Team"),
        )
        .with_columns(total_loss=pl.col("loss").cum_sum().over("Season", "Team"))
        .select(
            pl.all().sort_by(["Season", "Round.Number"]).over("Team").sort_by(["Team"])
        )
        .with_columns(
            pl.col("total_wins").shift().over(["Team", "Season"]).name.prefix("prev_")
        )
        .with_columns(pl.col("prev_total_wins").fill_null(0))
        .with_columns(
            pl.col("total_loss").shift().over(["Team", "Season"]).name.prefix("prev_")
        )
        .with_columns(pl.col("prev_total_loss").fill_null(0))
        .drop(["win", "loss"])
    )


def prepare_main_features(
    ladder: pl.DataFrame, total_win_loss: pl.DataFrame
) -> pl.DataFrame:
    """Join ladder standings with win/loss records and prepare main features."""
    return (
        ladder.select(
            pl.all().sort_by(["Season", "Round.Number"]).over("Team").sort_by(["Team"])
        )
        .with_columns(
            pl.col("Percentage").shift().over(["Team", "Season"]).name.prefix("prev_")
        )
        .with_columns(pl.col("prev_Percentage").fill_null(0))
        .select("Team", "Season", "Round.Number", "Percentage", "prev_Percentage")
        .join(total_win_loss, how="left", on=["Team", "Season", "Round.Number"])
    )


def prepare_feature_dataframe(
    results_df: pl.DataFrame, main_features: pl.DataFrame
) -> pl.DataFrame:
    """Join results with features for both home and away teams."""
    return (
        results_df.join(
            main_features,
            left_on=["Season", "Round.Number", "Home.Team"],
            right_on=["Season", "Round.Number", "Team"],
            how="left",
            coalesce=True,
        )
        .rename(
            {
                "Percentage": "home_team_percentage",
                "prev_Percentage": "home_team_prev_percentage",
                "total_wins": "home_total_wins",
                "total_loss": "home_total_loss",
                "prev_total_wins": "home_prev_total_wins",
                "prev_total_loss": "home_prev_total_loss",
            }
        )
        .join(
            main_features,
            left_on=["Season", "Round.Number", "Away.Team"],
            right_on=["Season", "Round.Number", "Team"],
            how="left",
            coalesce=True,
        )
        .rename(
            {
                "Percentage": "away_team_percentage",
                "prev_Percentage": "away_team_prev_percentage",
                "total_wins": "away_total_wins",
                "total_loss": "away_total_loss",
                "prev_total_wins": "away_prev_total_wins",
                "prev_total_loss": "away_prev_total_loss",
            }
        )
        .with_columns(
            prev_home_win_percentage=(
                pl.col("home_prev_total_wins")
                / (pl.col("home_prev_total_wins") + pl.col("home_prev_total_loss"))
            ).fill_nan(0),
            prev_away_win_percentage=(
                pl.col("away_prev_total_wins")
                / (pl.col("away_prev_total_wins") + pl.col("away_prev_total_loss"))
            ).fill_nan(0),
            home_win_percentage=(
                pl.col("home_total_wins")
                / (pl.col("home_total_wins") + pl.col("home_total_loss"))
            ).fill_nan(0),
            away_win_percentage=(
                pl.col("away_total_wins")
                / (pl.col("away_total_wins") + pl.col("away_total_loss"))
            ).fill_nan(0),
            prev_home_delta_wins=pl.col("home_prev_total_wins")
            - pl.col("away_prev_total_wins"),
            prev_home_delta_loss=pl.col("home_prev_total_loss")
            - pl.col("away_prev_total_loss"),
        )
        .with_columns(Round=pl.col("Round.Number") - 1)
    )


def split_train_test(
    feature_df: pl.DataFrame, predict_round: Tuple[int, int]
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Split data into training and test sets."""
    test_year = pl.col("Season") == predict_round[0]
    test_round = pl.col("Round") == predict_round[1]
    
    train_data = feature_df.filter(~(test_year & test_round))
    test_season_data = feature_df.filter(test_year)
    
    return train_data, test_season_data


def prepare_test_data(
    fixture: pl.DataFrame,
    test_season_data: pl.DataFrame,
    predict_round: Tuple[int, int],
) -> pl.DataFrame:
    """Create test dataset from fixture and final round stats."""
    # Extract home and away stats from test season
    test_home = test_season_data.select(
        "Home.Team",
        "home_team_percentage",
        "home_total_wins",
        "home_total_loss",
        "Round.Number",
    ).rename(
        {
            "Home.Team": "Team",
            "home_team_percentage": "percentage",
            "home_total_wins": "total_wins",
            "home_total_loss": "total_loss",
        }
    )

    test_away = test_season_data.select(
        "Away.Team",
        "away_team_percentage",
        "away_total_wins",
        "away_total_loss",
        "Round.Number",
    ).rename(
        {
            "Away.Team": "Team",
            "away_team_percentage": "percentage",
            "away_total_wins": "total_wins",
            "away_total_loss": "total_loss",
        }
    )

    # Get latest stats per team
    test_features_combined = pl.concat([test_home, test_away], how="vertical")
    test_features = (
        test_features_combined.group_by("Team")
        .agg(pl.col("Round.Number").max())
        .join(
            test_features_combined,
            how="inner",
            on=["Team", "Round.Number"],
        )
        .drop("Round.Number")
    )

    # Join with fixture for prediction round
    return (
        fixture.filter(pl.col("Round") == predict_round[1])
        .join(
            test_features, how="left", left_on="Home.Team", right_on="Team"
        )
        .rename(
            {
                "percentage": "home_team_prev_percentage",
                "total_wins": "home_prev_total_wins",
                "total_loss": "home_prev_total_loss",
            }
        )
        .join(
            test_features, how="left", left_on="Away.Team", right_on="Team"
        )
        .rename(
            {
                "percentage": "away_team_prev_percentage",
                "total_wins": "away_prev_total_wins",
                "total_loss": "away_prev_total_loss",
            }
        )
        .with_columns(
            prev_home_delta_wins=pl.col("home_prev_total_wins")
            - pl.col("away_prev_total_wins"),
            prev_home_delta_loss=pl.col("home_prev_total_loss")
            - pl.col("away_prev_total_loss"),
        )
    )


def setup_model_data(
    train_data: pl.DataFrame, test_data: pl.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, list]]:
    """Extract feature arrays and coordinates for the model."""
    x_percentage_train = train_data[
        "home_team_prev_percentage", "away_team_prev_percentage"
    ].to_numpy()
    x_delta_wins_train = train_data[
        "prev_home_delta_wins", "prev_home_delta_loss"
    ].to_numpy()
    y_train = train_data[["home_win"]].to_numpy()[:, 0]

    x_percentage_test = test_data[
        "home_team_prev_percentage", "away_team_prev_percentage"
    ].to_numpy()
    x_delta_wins_test = test_data[
        "prev_home_delta_wins", "prev_home_delta_loss"
    ].to_numpy()

    coords = {
        "perc_coeffs": ["home_team_prev_percentage", "away_team_prev_percentage"],
        "delta_wins_coeffs": ["prev_home_delta_wins", "prev_home_delta_loss"],
    }

    return (
        x_percentage_train,
        x_delta_wins_train,
        y_train,
        x_percentage_test,
        x_delta_wins_test,
        coords,
    )


def fit_bayesian_model(
    x_percentage_train: np.ndarray,
    x_delta_wins_train: np.ndarray,
    y_train: np.ndarray,
    coords: Dict[str, list],
    random_seed: int = 123,
) -> pm.model.Model:
    """Fit Bayesian logistic regression model."""
    with pm.Model(coords=coords) as model:
        # Data containers
        X_perc = pm.Data("X_perc", x_percentage_train)
        X_delta_wins = pm.Data("X_delta_wins", x_delta_wins_train)
        y = pm.Data("y", y_train)

        # Priors
        a = pm.Normal("a", mu=0, sigma=0.1)
        bp = pm.Normal("bp", mu=0, sigma=1, dims="perc_coeffs")
        bw = pm.Normal("bw", mu=0, sigma=1, dims="delta_wins_coeffs")

        # Linear model
        mu = a + pm.math.dot(X_perc, bp) + pm.math.dot(X_delta_wins, bw)
        # Link function
        p = pm.Deterministic("p", pm.math.invlogit(mu))
        # Likelihood
        pm.Bernoulli("obs", p=p, observed=y)

        # Fit the model
        idata = pm.sample(random_seed=random_seed, progressbar=True)

    return idata


def extract_parameters(
    idata: pm.model.Model,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Extract posterior means for model parameters."""
    a_param = az.summary(idata, var_names=["a"])["mean"]["a"]
    beta_p = np.array(az.summary(idata, var_names=["bp"])["mean"])
    beta_w = np.array(az.summary(idata, var_names=["bw"])["mean"])

    return a_param, beta_p, beta_w


def logistic(x: np.ndarray) -> np.ndarray:
    """Apply logistic function."""
    return 1 / (1 + np.exp(-x))


def generate_predictions(
    a_param: float,
    beta_p: np.ndarray,
    beta_w: np.ndarray,
    x_percentage_test: np.ndarray,
    x_delta_wins_test: np.ndarray,
) -> list:
    """Generate predictions for test data."""
    p_array = (
        a_param
        + np.dot(x_percentage_test, beta_p)
        + np.dot(x_delta_wins_test, beta_w)
    )

    result_array = [
        (
            logistic(p),
            1 + np.log2(logistic(p)),
            1 + np.log2(1 - logistic(p)),
        )
        for p in p_array
    ]

    return result_array


def main():
    """Execute the full pipeline."""
    print("Loading data...")
    ladder, results, fixture = load_data(
        DATA_CONFIG["ladder_path"],
        DATA_CONFIG["results_path"],
        DATA_CONFIG["fixture_path"],
    )

    print("Processing results...")
    results_df = process_results(results)

    print("Calculating win/loss records...")
    total_win_loss = calculate_win_loss_records(results_df)

    print("Preparing main features...")
    main_features = prepare_main_features(ladder, total_win_loss)

    print("Preparing feature dataframe...")
    feature_df = prepare_feature_dataframe(results_df, main_features)

    print("Splitting train/test data...")
    train_data, test_season_data = split_train_test(
        feature_df, DATA_CONFIG["predict_round"]
    )

    print("Preparing test data...")
    test_data = prepare_test_data(
        fixture, test_season_data, DATA_CONFIG["predict_round"]
    )

    print("Setting up model data...")
    (
        x_percentage_train,
        x_delta_wins_train,
        y_train,
        x_percentage_test,
        x_delta_wins_test,
        coords,
    ) = setup_model_data(train_data, test_data)

    print("Fitting Bayesian model (this may take a minute)...")
    idata = fit_bayesian_model(
        x_percentage_train, x_delta_wins_train, y_train, coords
    )

    print("Extracting parameters...")
    a_param, beta_p, beta_w = extract_parameters(idata)

    print("Generating predictions...")
    predictions = generate_predictions(
        a_param, beta_p, beta_w, x_percentage_test, x_delta_wins_test
    )

    print("\n" + "=" * 80)
    print(f"Predictions for Round {DATA_CONFIG['predict_round'][1]}, "
          f"Season {DATA_CONFIG['predict_round'][0]}")
    print("=" * 80)
    print("\nFixture:")
    print(test_data.select("Home.Team", "Away.Team"))
    print("\nPredictions (Probability of Home Win, Log Odds Home, Log Odds Away):")
    
    # Convert to dict for easier iteration
    test_data_dict = test_data.select("Home.Team", "Away.Team").to_dicts()
    for i, pred in enumerate(predictions):
        home_team = test_data_dict[i]["Home.Team"]
        away_team = test_data_dict[i]["Away.Team"]
        prob_home, log_odds_home, log_odds_away = pred
        print(
            f"{home_team:12} vs {away_team:12} | "
            f"P(Home)={prob_home:.3f} | "
            f"Log Odds: {log_odds_home:.3f} (H) / {log_odds_away:.3f} (A)"
        )


if __name__ == "__main__":
    main()
