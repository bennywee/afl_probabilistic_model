"""Data loading and preparation functions."""

import polars as pl
from typing import Tuple


def load_data(
    ladder_path: str, results_path: str, fixture_path: str
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load and perform initial transforms on parquet data.
    
    Args:
        ladder_path: Path to ladder standings parquet file
        results_path: Path to historical results parquet file
        fixture_path: Path to fixture parquet file
        
    Returns:
        Tuple of (ladder, results, fixture) DataFrames
    """
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
    """Add home_win column and filter to regular season only.
    
    Args:
        results: Results DataFrame
        
    Returns:
        Processed results DataFrame with home_win indicator
    """
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
    """Calculate cumulative wins/losses for each team.
    
    Args:
        results_df: Processed results DataFrame
        
    Returns:
        DataFrame with cumulative and previous round win/loss statistics
    """
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
    """Join ladder standings with win/loss records and prepare main features.
    
    Args:
        ladder: Ladder standings DataFrame
        total_win_loss: Cumulative win/loss DataFrame
        
    Returns:
        DataFrame with combined ladder and win/loss features
    """
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
    """Join results with features for both home and away teams.
    
    Args:
        results_df: Processed results DataFrame
        main_features: Main features DataFrame
        
    Returns:
        DataFrame with all features for modeling
    """
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
    feature_df: pl.DataFrame, predict_round: tuple
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Split data into training and test sets.
    
    Args:
        feature_df: Feature DataFrame
        predict_round: Tuple of (season, round) for prediction
        
    Returns:
        Tuple of (train_data, test_season_data)
    """
    test_year = pl.col("Season") == predict_round[0]
    test_round = pl.col("Round") == predict_round[1]
    
    train_data = feature_df.filter(~(test_year & test_round))
    test_season_data = feature_df.filter(test_year)
    
    return train_data, test_season_data


def add_games_played_features(
    results_df: pl.DataFrame,
    feature_df: pl.DataFrame,
) -> pl.DataFrame:
    """Add cumulative games-played features for home and away teams.
    
    Args:
        results_df: Processed results DataFrame with Season, Round.Number, Home.Team, Away.Team
        feature_df: Feature DataFrame to enrich
        
    Returns:
        Feature DataFrame with home_games_played and away_games_played columns
    """
    # Create games-played frame from all games
    all_games = pl.concat([
        results_df.select("Season", "Round.Number", "Home.Team").rename({"Home.Team": "Team"}),
        results_df.select("Season", "Round.Number", "Away.Team").rename({"Away.Team": "Team"}),
    ]).sort(["Season", "Team", "Round.Number"])
    
    games_played = (
        all_games
        .with_columns(
            games_count=pl.lit(1)
        )
        .with_columns(
            games_played=pl.col("games_count").cum_sum().over("Season", "Team")
        )
        .select("Season", "Round.Number", "Team", "games_played")
        # Subtract 1 to get games played BEFORE this round
        .with_columns(
            games_played=pl.col("games_played") - 1
        )
    )
    
    # Join back to feature_df for home team
    feature_df = feature_df.join(
        games_played,
        left_on=["Season", "Round.Number", "Home.Team"],
        right_on=["Season", "Round.Number", "Team"],
        how="left",
    ).rename({"games_played": "home_games_played"})
    
    # Join for away team
    feature_df = feature_df.join(
        games_played,
        left_on=["Season", "Round.Number", "Away.Team"],
        right_on=["Season", "Round.Number", "Team"],
        how="left",
    ).rename({"games_played": "away_games_played"})
    
    return feature_df


def prepare_test_data(
    fixture: pl.DataFrame,
    test_season_data: pl.DataFrame,
    predict_round: tuple,
) -> pl.DataFrame:
    """Create test dataset from fixture and final round stats.
    
    Args:
        fixture: Fixture DataFrame
        test_season_data: Test season data from feature_df
        predict_round: Tuple of (season, round) for prediction
        
    Returns:
        Test data with features for the prediction round
    """
    # Use the enriched `test_season_data` (which should already include
    # games-played columns added earlier by `add_games_played_features`).
    # Extract home and away stats from test season, including games_played
    test_home = (
        test_season_data.select(
            "Season",
            "Home.Team",
            "home_team_percentage",
            "home_total_wins",
            "home_total_loss",
            "Round.Number",
            "home_games_played",
        )
        .rename(
            {
                "Home.Team": "Team",
                "home_team_percentage": "percentage",
                "home_total_wins": "total_wins",
                "home_total_loss": "total_loss",
                "home_games_played": "games_played",
            }
        )
    )

    test_away = (
        test_season_data.select(
            "Season",
            "Away.Team",
            "away_team_percentage",
            "away_total_wins",
            "away_total_loss",
            "Round.Number",
            "away_games_played",
        )
        .rename(
            {
                "Away.Team": "Team",
                "away_team_percentage": "percentage",
                "away_total_wins": "total_wins",
                "away_total_loss": "total_loss",
                "away_games_played": "games_played",
            }
        )
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

    # Join with fixture for prediction round and include games-played as home/away columns
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
                "games_played": "home_games_played",
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
                "games_played": "away_games_played",
            }
        )
        .with_columns(
            prev_home_delta_wins=pl.col("home_prev_total_wins")
            - pl.col("away_prev_total_wins"),
            prev_home_delta_loss=pl.col("home_prev_total_loss")
            - pl.col("away_prev_total_loss"),
        )
    )
