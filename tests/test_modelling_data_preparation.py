"""Tests for src.modelling.data_preparation module."""

import pytest
import polars as pl
import numpy as np
from src.modelling.data_preparation import (
    load_data,
    process_results,
    calculate_win_loss_records,
    prepare_main_features,
    prepare_feature_dataframe,
    split_train_test,
    prepare_test_data,
)


@pytest.fixture
def sample_results_df():
    """Create a minimal results DataFrame for testing."""
    return pl.DataFrame({
        "Season": [2020, 2020, 2020, 2020],
        "Round.Number": [1, 1, 2, 2],
        "Home.Points": [100, 80, 95, 110],
        "Away.Points": [90, 85, 95, 100],
        "Home.Team": ["TeamA", "TeamC", "TeamA", "TeamB"],
        "Away.Team": ["TeamB", "TeamD", "TeamC", "TeamD"],
        "Round": ["R1", "R1", "R2", "R2"],
    })


@pytest.fixture
def sample_ladder_df():
    """Create a minimal ladder DataFrame for testing."""
    return pl.DataFrame({
        "Team": ["TeamA", "TeamB", "TeamA", "TeamB"],
        "Season": [2020, 2020, 2020, 2020],
        "Round.Number": [1, 1, 2, 2],
        "Percentage": [0.600, 0.500, 0.650, 0.550],
    })


@pytest.fixture
def sample_fixture_df():
    """Create a minimal fixture DataFrame for testing."""
    return pl.DataFrame({
        "Round": [28, 28, 29, 29],
        "Home.Team": ["TeamA", "TeamC", "TeamA", "TeamB"],
        "Away.Team": ["TeamB", "TeamD", "TeamC", "TeamD"],
    })


class TestProcessResults:
    """Tests for process_results function."""

    def test_process_results_adds_home_win_column(self, sample_results_df):
        """Test that home_win column is added correctly."""
        result = process_results(sample_results_df)
        assert "home_win" in result.columns
        assert result.shape[0] == 4

    def test_process_results_correct_wins(self, sample_results_df):
        """Test that home_win values are correct."""
        result = process_results(sample_results_df)
        home_wins = result["home_win"].to_list()
        # TeamA: 100 > 90 = 1, TeamC: 80 < 85 = 0, TeamA: 95 = 95 = 0, TeamB: 110 > 100 = 1
        assert home_wins == [1, 0, 0, 1]

    def test_process_results_filters_playoffs(self):
        """Test that playoff rounds are filtered out."""
        df = pl.DataFrame({
            "Season": [2020, 2020, 2020],
            "Round.Number": [1, 2, 3],
            "Home.Points": [100, 100, 100],
            "Away.Points": [90, 90, 90],
            "Home.Team": ["A", "A", "A"],
            "Away.Team": ["B", "B", "B"],
            "Round": ["R1", "QF", "GF"],
        })
        result = process_results(df)
        # Should only have the regular season match
        assert result.shape[0] == 1
        assert result["Round.Number"][0] == 1

    def test_process_results_selects_correct_columns(self, sample_results_df):
        """Test that only correct columns are selected."""
        result = process_results(sample_results_df)
        expected_cols = {"Season", "Round.Number", "home_win", "Home.Team", "Away.Team"}
        assert set(result.columns) == expected_cols


class TestCalculateWinLossRecords:
    """Tests for calculate_win_loss_records function."""

    def test_calculates_cumulative_wins(self, sample_results_df):
        """Test cumulative win calculation."""
        results_df = process_results(sample_results_df)
        result = calculate_win_loss_records(results_df)
        
        assert "total_wins" in result.columns
        assert "total_loss" in result.columns
        assert "prev_total_wins" in result.columns
        assert "prev_total_loss" in result.columns

    def test_separates_home_and_away(self, sample_results_df):
        """Test that home and away teams are separated."""
        results_df = process_results(sample_results_df)
        result = calculate_win_loss_records(results_df)
        
        # Should have 2 rows per match (home and away perspective)
        assert result.shape[0] == 8  # 4 matches * 2 perspectives

    def test_previous_round_stats_null_filled(self, sample_results_df):
        """Test that first round previous stats are filled with 0."""
        results_df = process_results(sample_results_df)
        result = calculate_win_loss_records(results_df)
        
        # First round should have 0 for previous wins/losses
        first_round = result.filter(pl.col("Round.Number") == 1)
        assert (first_round["prev_total_wins"] == 0).all()
        assert (first_round["prev_total_loss"] == 0).all()


class TestPrepareMainFeatures:
    """Tests for prepare_main_features function."""

    def test_returns_dataframe_with_percentage(self, sample_ladder_df):
        """Test that percentage columns are preserved."""
        win_loss = calculate_win_loss_records(pl.DataFrame({
            "Season": [2020, 2020, 2020, 2020],
            "Round.Number": [1, 1, 2, 2],
            "Team": ["TeamA", "TeamB", "TeamA", "TeamB"],
            "total_wins": [1, 0, 1, 1],
            "total_loss": [0, 1, 0, 1],
            "prev_total_wins": [0, 0, 1, 0],
            "prev_total_loss": [0, 0, 0, 1],
        }))
        
        result = prepare_main_features(sample_ladder_df, win_loss)
        assert "Percentage" in result.columns
        assert "prev_Percentage" in result.columns


class TestPrepareFeatureDataframe:
    """Tests for prepare_feature_dataframe function."""

    def test_feature_df_contains_home_and_away_stats(self, sample_results_df):
        """Test that feature dataframe has both home and away team stats."""
        results_df = process_results(sample_results_df)
        win_loss = calculate_win_loss_records(results_df)
        
        # Create mock main features
        main_features = pl.DataFrame({
            "Team": ["TeamA", "TeamB", "TeamC", "TeamD"],
            "Season": [2020] * 4,
            "Round.Number": [1] * 4,
            "Percentage": [0.6, 0.5, 0.5, 0.4],
            "prev_Percentage": [0.0, 0.0, 0.0, 0.0],
            "total_wins": [1, 0, 0, 0],
            "total_loss": [0, 1, 1, 1],
            "prev_total_wins": [0, 0, 0, 0],
            "prev_total_loss": [0, 0, 0, 0],
        })
        
        result = prepare_feature_dataframe(results_df, main_features)
        
        # Check home team columns exist
        assert any("home_team" in col for col in result.columns)
        assert any("away_team" in col for col in result.columns)


class TestSplitTrainTest:
    """Tests for split_train_test function."""

    def test_splits_correctly(self, sample_results_df):
        """Test that data is split correctly by round."""
        results_df = process_results(sample_results_df)
        win_loss = calculate_win_loss_records(results_df)
        main_features = prepare_main_features(sample_ladder_df(), win_loss)
        
        feature_df = pl.DataFrame({
            "Season": [2020, 2020, 2025, 2025],
            "Round.Number": [1, 2, 28, 29],
            "Round": [0, 1, 27, 28],
            "home_win": [1, 0, 1, 0],
        })
        
        train_data, test_season_data = split_train_test(feature_df, (2025, 29))
        
        # Test season (2025) and round 29 should be in test season but not train
        assert train_data.filter((pl.col("Season") == 2025) & (pl.col("Round") == 28)).shape[0] == 1
        assert test_season_data.filter(pl.col("Season") == 2025).shape[0] == 2

    def test_train_excludes_predict_round(self):
        """Test that prediction round is excluded from training."""
        feature_df = pl.DataFrame({
            "Season": [2025, 2025, 2025],
            "Round": [27, 28, 29],
            "home_win": [1, 0, 1],
        })
        
        train_data, test_data = split_train_test(feature_df, (2025, 29))
        
        # Prediction round should not be in training
        assert train_data.filter(pl.col("Round") == 29).shape[0] == 0


class TestPrepareTestData:
    """Tests for prepare_test_data function."""

    def test_returns_dataframe(self):
        """Test that prepare_test_data returns a DataFrame."""
        fixture = pl.DataFrame({
            "Round": [29, 29],
            "Home.Team": ["TeamA", "TeamC"],
            "Away.Team": ["TeamB", "TeamD"],
        })
        
        test_season_data = pl.DataFrame({
            "Season": [2025, 2025, 2025, 2025],
            "Round.Number": [28, 28, 28, 28],
            "Home.Team": ["TeamA", "TeamC", "TeamC", "TeamA"],
            "Away.Team": ["TeamB", "TeamD", "TeamA", "TeamD"],
            "home_team_percentage": [0.6, 0.5, 0.5, 0.6],
            "away_team_percentage": [0.4, 0.4, 0.6, 0.4],
            "home_total_wins": [20, 18, 18, 20],
            "home_total_loss": [8, 10, 10, 8],
            "away_total_wins": [16, 16, 20, 16],
            "away_total_loss": [12, 12, 8, 12],
        })
        
        result = prepare_test_data(fixture, test_season_data, (2025, 29))
        
        assert isinstance(result, pl.DataFrame)
        assert "prev_home_delta_wins" in result.columns
        assert "prev_home_delta_loss" in result.columns
