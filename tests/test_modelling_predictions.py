"""Tests for src.modelling.predictions module."""

import numpy as np
import polars as pl
from src.modelling.predictions import (
    logistic,
    generate_predictions,
    print_predictions,
)


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
