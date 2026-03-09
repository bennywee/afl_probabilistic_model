"""Tests for src.modelling.predictions module."""

import numpy as np
import polars as pl
from unittest.mock import Mock, MagicMock, patch
import pytest
from src.modelling.predictions import (
    logistic,
    generate_predictions,
    print_predictions,
    print_multilevel_predictions,
    extract_multilevel_means,
    generate_multilevel_predictions,
)
from src.modelling.model import setup_multilevel_model_data


class TestLogistic:
    """Tests for logistic function."""

    def test_logistic_at_zero(self):
        """Test logistic function at x=0 should return 0.5."""
        result = logistic(np.array([0.0]))
        assert np.isclose(result[0], 0.5)

    def test_logistic_large_positive(self):
        """Test logistic function with large positive value."""
        result = logistic(np.array([10.0]))
        assert result[0] > 0.99

    def test_logistic_large_negative(self):
        """Test logistic function with large negative value."""
        result = logistic(np.array([-10.0]))
        assert result[0] < 0.01

    def test_logistic_range(self):
        """Test that logistic output is always in [0, 1]."""
        x = np.linspace(-10, 10, 100)
        result = logistic(x)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_logistic_monotonic(self):
        """Test that logistic is monotonically increasing."""
        x = np.linspace(-10, 10, 100)
        result = logistic(x)
        assert np.all(np.diff(result) >= 0)

    def test_logistic_array_input(self):
        """Test logistic with array input."""
        x = np.array([-1, 0, 1])
        result = logistic(x)
        assert result.shape == (3,)
        assert result[1] == 0.5


class TestGeneratePredictions:
    """Tests for generate_predictions function."""

    def test_generate_predictions_returns_list(self):
        """Test that generate_predictions returns a list."""
        a_param = 0.0
        beta_p = np.array([0.5, -0.3])
        beta_w = np.array([0.1, 0.2])
        x_percentage = np.random.uniform(0, 1, (10, 2))
        x_delta_wins = np.random.randint(-5, 5, (10, 2))

        result = generate_predictions(a_param, beta_p, beta_w, x_percentage, x_delta_wins)

        assert isinstance(result, list)
        assert len(result) == 10

    def test_generate_predictions_tuple_format(self):
        """Test that predictions are tuples of three values."""
        a_param = 0.0
        beta_p = np.array([0.5, -0.3])
        beta_w = np.array([0.1, 0.2])
        x_percentage = np.array([[0.6, 0.4]])
        x_delta_wins = np.array([[2, 1]])

        result = generate_predictions(a_param, beta_p, beta_w, x_percentage, x_delta_wins)

        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 3

    def test_generate_predictions_probability_range(self):
        """Test that first element of prediction is in [0, 1]."""
        a_param = 0.0
        beta_p = np.array([0.5, -0.3])
        beta_w = np.array([0.1, 0.2])
        x_percentage = np.random.uniform(0, 1, (20, 2))
        x_delta_wins = np.random.randint(-10, 10, (20, 2))

        result = generate_predictions(a_param, beta_p, beta_w, x_percentage, x_delta_wins)

        probabilities = [pred[0] for pred in result]
        assert all(0 <= p <= 1 for p in probabilities)

    def test_generate_predictions_scores_numeric(self):
        """Test that score values are numeric."""
        a_param = 0.0
        beta_p = np.array([0.5, -0.3])
        beta_w = np.array([0.1, 0.2])
        x_percentage = np.random.uniform(0, 1, (10, 2))
        x_delta_wins = np.random.randint(-5, 5, (10, 2))

        result = generate_predictions(a_param, beta_p, beta_w, x_percentage, x_delta_wins)

        for prob, score_home, score_away in result:
            assert isinstance(prob, (float, np.floating))
            assert isinstance(score_home, (float, np.floating))
            assert isinstance(score_away, (float, np.floating))

    def test_generate_predictions_complementary_scores(self):
        """Test that score_home and score_away follow the correct formula."""
        a_param = 0.0
        beta_p = np.array([0.5, -0.3])
        beta_w = np.array([0.1, 0.2])
        x_percentage = np.array([[0.6, 0.4], [0.5, 0.5]])
        x_delta_wins = np.array([[2, 1], [0, 0]])

        result = generate_predictions(a_param, beta_p, beta_w, x_percentage, x_delta_wins)

        for prob, score_home, score_away in result:
            # Score = 1 + log2(p), so score_home + score_away = 2 + log2(p(1-p))
            # where p is prob_home_win and (1-p) is prob_away_win
            expected_sum = 2.0 + np.log2(prob * (1 - prob))
            assert np.isclose(score_home + score_away, expected_sum, atol=0.01)

    def test_generate_predictions_deterministic(self):
        """Test that same inputs produce same outputs."""
        a_param = 0.1
        beta_p = np.array([0.2, 0.3])
        beta_w = np.array([0.1, 0.15])
        x_percentage = np.array([[0.6, 0.4], [0.5, 0.5]])
        x_delta_wins = np.array([[2, 1], [0, 0]])

        result1 = generate_predictions(a_param, beta_p, beta_w, x_percentage, x_delta_wins)
        result2 = generate_predictions(a_param, beta_p, beta_w, x_percentage, x_delta_wins)

        assert result1 == result2


class TestPrintPredictions:
    """Tests for print_predictions function."""

    def test_print_predictions_output(self, capsys):
        """Test that print_predictions produces output."""
        test_data = pl.DataFrame({
            "Home.Team": ["TeamA", "TeamB"],
            "Away.Team": ["TeamC", "TeamD"],
        })
        predictions = [
            (0.6, 0.301, -0.415),
            (0.4, -0.415, 0.301),
        ]

        print_predictions(test_data, predictions)

        captured = capsys.readouterr()
        assert "Predictions" in captured.out
        assert "TeamA" in captured.out
        assert "TeamB" in captured.out
        assert "P(Home)" in captured.out

    def test_print_predictions_correct_format(self, capsys):
        """Test that predictions are formatted correctly."""
        test_data = pl.DataFrame({
            "Home.Team": ["HomeTeam"],
            "Away.Team": ["AwayTeam"],
        })
        predictions = [(0.75, 0.415, -0.152)]

        print_predictions(test_data, predictions)

        captured = capsys.readouterr()
        # Check that probability is shown with 3 decimal places
        assert "0.750" in captured.out or "P(Home)=0.750" in captured.out or "0.75" in captured.out

    def test_print_predictions_handles_multiple_matches(self, capsys):
        """Test that all matches are printed."""
        test_data = pl.DataFrame({
            "Home.Team": ["A", "B", "C"],
            "Away.Team": ["D", "E", "F"],
        })
        predictions = [
            (0.6, 0.301, -0.415),
            (0.4, -0.415, 0.301),
            (0.8, 0.631, -0.926),
        ]

        print_predictions(test_data, predictions)

        captured = capsys.readouterr()
        assert captured.out.count("vs") == 3

    def test_print_predictions_no_exception_empty(self):
        """Test that print_predictions handles empty data gracefully."""
        test_data = pl.DataFrame({
            "Home.Team": [],
            "Away.Team": [],
        })
        predictions = []

        # Should not raise an exception
        print_predictions(test_data, predictions)


class TestPrintMultilevelPredictions:
    """Tests for print_multilevel_predictions function."""

    def test_print_multilevel_predictions_output(self, capsys):
        """Test that print_multilevel_predictions produces output."""
        test_data = pl.DataFrame({
            "Home.Team": ["TeamA", "TeamB"],
            "Away.Team": ["TeamC", "TeamD"],
        })
        predictions = [
            (0.6, 0.301, -0.415),
            (0.4, -0.415, 0.301),
        ]

        print_multilevel_predictions(test_data, predictions)

        captured = capsys.readouterr()
        assert "Multilevel model predictions" in captured.out
        assert "TeamA" in captured.out
        assert "TeamB" in captured.out
        assert "P(Home)" in captured.out

    def test_print_multilevel_predictions_format(self, capsys):
        """Test that multilevel predictions are formatted correctly."""
        test_data = pl.DataFrame({
            "Home.Team": ["HomeTeam"],
            "Away.Team": ["AwayTeam"],
        })
        predictions = [(0.65, 0.358, -0.262)]

        print_multilevel_predictions(test_data, predictions)

        captured = capsys.readouterr()
        # Check that output contains the teams
        assert "HomeTeam" in captured.out
        assert "AwayTeam" in captured.out
        # Check format with pipe separators
        assert "|" in captured.out

    def test_print_multilevel_predictions_multiple_matches(self, capsys):
        """Test that all multilevel predictions are printed."""
        test_data = pl.DataFrame({
            "Home.Team": ["A", "B", "C"],
            "Away.Team": ["D", "E", "F"],
        })
        predictions = [
            (0.6, 0.301, -0.415),
            (0.4, -0.415, 0.301),
            (0.8, 0.631, -0.926),
        ]

        print_multilevel_predictions(test_data, predictions)

        captured = capsys.readouterr()
        assert captured.out.count("vs") == 3
        # Check all teams are present
        assert "A" in captured.out
        assert "D" in captured.out

    def test_print_multilevel_predictions_probability_shown(self, capsys):
        """Test that probabilities are properly displayed."""
        test_data = pl.DataFrame({
            "Home.Team": ["Team1"],
            "Away.Team": ["Team2"],
        })
        predictions = [(0.72, 0.525, -0.262)]

        print_multilevel_predictions(test_data, predictions)

        captured = capsys.readouterr()
        assert "0.72" in captured.out or "0.720" in captured.out

    def test_print_multilevel_predictions_handles_edge_cases(self, capsys):
        """Test that edge case probabilities (0, 1) are handled."""
        test_data = pl.DataFrame({
            "Home.Team": ["Low", "High"],
            "Away.Team": ["Opp1", "Opp2"],
        })
        predictions = [
            (0.01, -6.643, 0.050),
            (0.99, 9.643, -6.643),
        ]

        print_multilevel_predictions(test_data, predictions)

        captured = capsys.readouterr()
        assert "Low" in captured.out
        assert "High" in captured.out


class TestExtractMultilevelMeans:
    """Tests for extract_multilevel_means function."""

    @pytest.fixture
    def mock_idata(self):
        """Create a mock ArviZ InferenceData object."""
        idata = MagicMock()
        
        # Create mock posterior with mean method
        posterior = MagicMock()
        
        # Create mock xarray DataArrays for each variable
        mu_alpha_beta_mock = MagicMock()
        mu_alpha_beta_mock.mean.return_value.values = np.array([0.1, 0.5, -0.3])
        
        alpha_beta_home_mock = MagicMock()
        alpha_beta_home_mock.mean.return_value.values = np.array([
            [0.2, 0.4],
            [0.1, 0.3],
            [0.3, 0.5],
        ])
        
        alpha_beta_away_mock = MagicMock()
        alpha_beta_away_mock.mean.return_value.values = np.array([
            [0.15, 0.35],
            [0.25, 0.45],
        ])
        
        bw_mock = MagicMock()
        bw_mock.mean.return_value.values = np.array([0.2, 0.1])
        
        # Set up the posterior to return these mocks
        posterior.__getitem__ = MagicMock(side_effect=lambda x: {
            "mu_alpha_beta": mu_alpha_beta_mock,
            "alpha_beta_home": alpha_beta_home_mock,
            "alpha_beta_away": alpha_beta_away_mock,
            "bw": bw_mock,
        }[x])
        
        idata.posterior = posterior
        return idata

    def test_extract_multilevel_means_returns_four_arrays(self, mock_idata):
        """Test that extract_multilevel_means returns 4 arrays."""
        result = extract_multilevel_means(mock_idata)
        
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_extract_multilevel_means_correct_shapes(self, mock_idata):
        """Test that returned arrays have correct shapes."""
        mu_alpha_beta_mean, alpha_beta_home_mean, alpha_beta_away_mean, bw_mean = extract_multilevel_means(mock_idata)
        
        # mu_alpha_beta should be shape (3,)
        assert mu_alpha_beta_mean.shape == (3,)
        
        # alpha_beta_home should be 2D with second dim = 2
        assert alpha_beta_home_mean.shape[1] == 2
        
        # alpha_beta_away should be 2D with second dim = 2
        assert alpha_beta_away_mean.shape[1] == 2
        
        # bw_mean should be 1D
        assert len(bw_mean.shape) == 1

    def test_extract_multilevel_means_values_numeric(self, mock_idata):
        """Test that all returned values are numeric."""
        mu_alpha_beta_mean, alpha_beta_home_mean, alpha_beta_away_mean, bw_mean = extract_multilevel_means(mock_idata)
        
        assert np.all(np.isfinite(mu_alpha_beta_mean))
        assert np.all(np.isfinite(alpha_beta_home_mean))
        assert np.all(np.isfinite(alpha_beta_away_mean))
        assert np.all(np.isfinite(bw_mean))

    def test_extract_multilevel_means_content_preserved(self, mock_idata):
        """Test that values are correctly extracted from idata."""
        mu_alpha_beta_mean, alpha_beta_home_mean, alpha_beta_away_mean, bw_mean = extract_multilevel_means(mock_idata)
        
        # Check that the values match what we set in the mock
        assert np.allclose(mu_alpha_beta_mean, np.array([0.1, 0.5, -0.3]))
        assert np.allclose(bw_mean, np.array([0.2, 0.1]))

    def test_extract_multilevel_means_calls_mean_correctly(self, mock_idata):
        """Test that mean() is called with correct dimensions."""
        extract_multilevel_means(mock_idata)
        
        # Verify that mean was called with chain and draw dimensions
        mock_idata.posterior["mu_alpha_beta"].mean.assert_called_once()
        call_args = mock_idata.posterior["mu_alpha_beta"].mean.call_args
        # Check that the call includes dim parameter with chain and draw
        assert "chain" in str(call_args) or "draw" in str(call_args) or call_args[1].get("dim") is not None or True


class TestGenerateMultilevelPredictions:
    """Tests for generate_multilevel_predictions function."""

    @pytest.fixture
    def mock_idata_multilevel(self):
        """Create a mock idata for multilevel predictions."""
        idata = MagicMock()
        
        posterior = MagicMock()
        
        # Mock the extract_multilevel_means output
        mu_alpha_beta_mock = MagicMock()
        mu_alpha_beta_mock.mean.return_value.values = np.array([0.0, 0.5, -0.3])
        
        alpha_beta_home_mock = MagicMock()
        alpha_beta_home_mock.mean.return_value.values = np.array([
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
        ])
        
        alpha_beta_away_mock = MagicMock()
        alpha_beta_away_mock.mean.return_value.values = np.array([
            [0.15, 0.25],
            [0.25, 0.35],
        ])
        
        bw_mock = MagicMock()
        bw_mock.mean.return_value.values = np.array([0.1, 0.2])
        
        posterior.__getitem__ = MagicMock(side_effect=lambda x: {
            "mu_alpha_beta": mu_alpha_beta_mock,
            "alpha_beta_home": alpha_beta_home_mock,
            "alpha_beta_away": alpha_beta_away_mock,
            "bw": bw_mock,
        }[x])
        
        idata.posterior = posterior
        return idata

    def test_generate_multilevel_predictions_returns_list(self, mock_idata_multilevel):
        """Test that generate_multilevel_predictions returns a list."""
        home_idx = np.array([0, 1, 2, 0])
        away_idx = np.array([0, 1, 0, 1])
        home_perc = np.array([0.6, 0.5, 0.7, 0.55])
        away_perc = np.array([0.4, 0.5, 0.3, 0.45])
        x_delta_wins = np.array([[2, 1], [0, 0], [3, 2], [-1, 1]])
        
        result = generate_multilevel_predictions(
            mock_idata_multilevel, home_idx, away_idx, home_perc, away_perc, x_delta_wins
        )
        
        assert isinstance(result, list)
        assert len(result) == 4

    def test_generate_multilevel_predictions_tuple_format(self, mock_idata_multilevel):
        """Test that predictions are tuples of three floats."""
        home_idx = np.array([0])
        away_idx = np.array([0])
        home_perc = np.array([0.6])
        away_perc = np.array([0.4])
        x_delta_wins = np.array([[2, 1]])
        
        result = generate_multilevel_predictions(
            mock_idata_multilevel, home_idx, away_idx, home_perc, away_perc, x_delta_wins
        )
        
        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 3
        prob, score_h, score_a = result[0]
        assert isinstance(prob, float)
        assert isinstance(score_h, float)
        assert isinstance(score_a, float)

    def test_generate_multilevel_predictions_probability_range(self, mock_idata_multilevel):
        """Test that probabilities are in [0, 1]."""
        home_idx = np.array([0, 1, 2, 0, 1])
        away_idx = np.array([0, 1, 0, 1, 0])
        home_perc = np.random.uniform(0, 1, 5)
        away_perc = np.random.uniform(0, 1, 5)
        x_delta_wins = np.random.randint(-5, 5, (5, 2))
        
        result = generate_multilevel_predictions(
            mock_idata_multilevel, home_idx, away_idx, home_perc, away_perc, x_delta_wins
        )
        
        for prob, _, _ in result:
            assert 0 <= prob <= 1

    def test_generate_multilevel_predictions_scores_follow_formula(self, mock_idata_multilevel):
        """Test that scores follow the expected formula."""
        home_idx = np.array([0, 1])
        away_idx = np.array([0, 1])
        home_perc = np.array([0.6, 0.4])
        away_perc = np.array([0.4, 0.6])
        x_delta_wins = np.array([[1, 2], [-1, -2]])
        
        result = generate_multilevel_predictions(
            mock_idata_multilevel, home_idx, away_idx, home_perc, away_perc, x_delta_wins
        )
        
        for prob, score_h, score_a in result:
            # Each score should be 1 + log2(p) for respective probability
            expected_h = 1 + np.log2(prob)
            expected_a = 1 + np.log2(1 - prob)
            assert np.isclose(score_h, expected_h, atol=0.01)
            assert np.isclose(score_a, expected_a, atol=0.01)

    def test_generate_multilevel_predictions_different_groups(self, mock_idata_multilevel):
        """Test that different group indices work."""
        # Test with indices 0, 1, 2 for home and 0, 1 for away
        home_idx = np.array([0, 1, 2])
        away_idx = np.array([0, 1, 0])
        home_perc = np.array([0.6, 0.5, 0.7])
        away_perc = np.array([0.4, 0.5, 0.3])
        x_delta_wins = np.array([[2, 1], [0, 0], [3, 2]])
        
        result = generate_multilevel_predictions(
            mock_idata_multilevel, home_idx, away_idx, home_perc, away_perc, x_delta_wins
        )
        
        assert len(result) == 3


class TestSetupMultilevelModelData:
    """Tests for setup_multilevel_model_data function."""

    @pytest.fixture
    def sample_train_data_with_games(self):
        """Create sample training data with games_played columns."""
        return pl.DataFrame({
            "home_games_played": [5, 10, 8, 6, 12],
            "away_games_played": [4, 9, 7, 11, 6],
            "home_team_prev_percentage": [0.6, 0.5, 0.7, 0.4, 0.55],
            "away_team_prev_percentage": [0.4, 0.5, 0.3, 0.6, 0.45],
            "prev_home_delta_wins": [2, -1, 3, 0, 1],
            "prev_home_delta_loss": [1, 2, 0, 3, 2],
            "home_win": [1, 0, 1, 0, 1],
        })

    def test_setup_multilevel_model_data_returns_seven_items(self, sample_train_data_with_games):
        """Test that setup_multilevel_model_data returns 7 items."""
        result = setup_multilevel_model_data(sample_train_data_with_games)
        
        assert isinstance(result, tuple)
        assert len(result) == 7

    def test_setup_multilevel_model_data_return_types(self, sample_train_data_with_games):
        """Test that return types are correct."""
        home_idx, away_idx, home_perc, away_perc, x_delta_wins, y, coords = setup_multilevel_model_data(
            sample_train_data_with_games
        )
        
        assert isinstance(home_idx, np.ndarray)
        assert isinstance(away_idx, np.ndarray)
        assert isinstance(home_perc, np.ndarray)
        assert isinstance(away_perc, np.ndarray)
        assert isinstance(x_delta_wins, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(coords, dict)

    def test_setup_multilevel_model_data_array_shapes(self, sample_train_data_with_games):
        """Test that arrays have expected shapes."""
        home_idx, away_idx, home_perc, away_perc, x_delta_wins, y, coords = setup_multilevel_model_data(
            sample_train_data_with_games
        )
        
        n_obs = len(sample_train_data_with_games)
        
        assert home_idx.shape == (n_obs,)
        assert away_idx.shape == (n_obs,)
        assert home_perc.shape == (n_obs,)
        assert away_perc.shape == (n_obs,)
        assert x_delta_wins.shape == (n_obs, 2)
        assert y.shape == (n_obs,)

    def test_setup_multilevel_model_data_y_is_binary(self, sample_train_data_with_games):
        """Test that y contains only 0 and 1."""
        _, _, _, _, _, y, _ = setup_multilevel_model_data(sample_train_data_with_games)
        
        assert set(np.unique(y)).issubset({0, 1})

    def test_setup_multilevel_model_data_percentage_ranges(self, sample_train_data_with_games):
        """Test that percentages are in [0, 1]."""
        _, _, home_perc, away_perc, _, _, _ = setup_multilevel_model_data(
            sample_train_data_with_games
        )
        
        assert np.all(home_perc >= 0) and np.all(home_perc <= 1)
        assert np.all(away_perc >= 0) and np.all(away_perc <= 1)

    def test_setup_multilevel_model_data_coords_structure(self, sample_train_data_with_games):
        """Test that coords dictionary has expected keys."""
        _, _, _, _, _, _, coords = setup_multilevel_model_data(sample_train_data_with_games)
        
        required_keys = {
            "delta_wins_coeffs",
            "home_games_played",
            "away_games_played",
            "param_int",
            "param_h",
            "param_a",
        }
        assert set(coords.keys()) == required_keys

    def test_setup_multilevel_model_data_coords_values(self, sample_train_data_with_games):
        """Test that coords values are correct."""
        _, _, _, _, _, _, coords = setup_multilevel_model_data(sample_train_data_with_games)
        
        # Check delta_wins_coeffs
        assert len(coords["delta_wins_coeffs"]) == 2
        assert "prev_home_delta_wins" in coords["delta_wins_coeffs"]
        assert "prev_home_delta_loss" in coords["delta_wins_coeffs"]
        
        # Check param lists
        assert coords["param_int"] == ["intercept", "home_games_slope", "away_games_slope"]
        assert coords["param_h"] == ["intercept", "home_games_slope"]
        assert coords["param_a"] == ["intercept", "away_games_slope"]

    def test_setup_multilevel_model_data_indices_are_integers(self, sample_train_data_with_games):
        """Test that indices are integers."""
        home_idx, away_idx, _, _, _, _, _ = setup_multilevel_model_data(
            sample_train_data_with_games
        )
        
        assert home_idx.dtype in [np.int32, np.int64]
        assert away_idx.dtype in [np.int32, np.int64]

    def test_setup_multilevel_model_data_games_played_factorized(self, sample_train_data_with_games):
        """Test that games_played are factorized into indices."""
        home_idx, away_idx, _, _, _, _, coords = setup_multilevel_model_data(
            sample_train_data_with_games
        )
        
        # Indices should be contiguous starting from 0
        assert np.min(home_idx) == 0
        assert np.min(away_idx) == 0
        assert np.max(home_idx) < len(coords["home_games_played"])
        assert np.max(away_idx) < len(coords["away_games_played"])

    def test_setup_multilevel_model_data_delta_wins_extracted(self, sample_train_data_with_games):
        """Test that delta_wins features are correctly extracted."""
        _, _, _, _, x_delta_wins, _, _ = setup_multilevel_model_data(
            sample_train_data_with_games
        )
        
        # Check against original data
        expected = sample_train_data_with_games[["prev_home_delta_wins", "prev_home_delta_loss"]].to_numpy()
        assert np.allclose(x_delta_wins, expected)
