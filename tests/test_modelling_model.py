"""Tests for src.modelling.model module."""

import pytest
import numpy as np
import polars as pl
from src.modelling.model import (
    setup_model_data,
    fit_bayesian_model,
    extract_parameters,
)


@pytest.fixture
def sample_train_data():
    """Create sample training data."""
    return pl.DataFrame({
        "home_team_prev_percentage": np.random.uniform(0, 1, 100),
        "away_team_prev_percentage": np.random.uniform(0, 1, 100),
        "prev_home_delta_wins": np.random.randint(-10, 10, 100),
        "prev_home_delta_loss": np.random.randint(-10, 10, 100),
        "home_win": np.random.randint(0, 2, 100),
    })


@pytest.fixture
def sample_test_data():
    """Create sample test data."""
    return pl.DataFrame({
        "home_team_prev_percentage": np.random.uniform(0, 1, 20),
        "away_team_prev_percentage": np.random.uniform(0, 1, 20),
        "prev_home_delta_wins": np.random.randint(-10, 10, 20),
        "prev_home_delta_loss": np.random.randint(-10, 10, 20),
    })


class TestSetupModelData:
    """Tests for setup_model_data function."""

    def test_setup_model_data_returns_correct_shapes(self, sample_train_data, sample_test_data):
        """Test that setup_model_data returns arrays with correct shapes."""
        (
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            x_percentage_test,
            x_delta_wins_test,
            coords,
        ) = setup_model_data(sample_train_data, sample_test_data)

        assert x_percentage_train.shape == (100, 2)
        assert x_delta_wins_train.shape == (100, 2)
        assert y_train.shape == (100,)
        assert x_percentage_test.shape == (20, 2)
        assert x_delta_wins_test.shape == (20, 2)

    def test_setup_model_data_returns_coords(self, sample_train_data, sample_test_data):
        """Test that coordinates dictionary is properly structured."""
        (
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            x_percentage_test,
            x_delta_wins_test,
            coords,
        ) = setup_model_data(sample_train_data, sample_test_data)

        assert isinstance(coords, dict)
        assert "perc_coeffs" in coords
        assert "delta_wins_coeffs" in coords
        assert len(coords["perc_coeffs"]) == 2
        assert len(coords["delta_wins_coeffs"]) == 2

    def test_setup_model_data_y_is_binary(self, sample_train_data, sample_test_data):
        """Test that y_train contains only 0 and 1 values."""
        (
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            x_percentage_test,
            x_delta_wins_test,
            coords,
        ) = setup_model_data(sample_train_data, sample_test_data)

        assert set(np.unique(y_train)).issubset({0, 1})

    def test_setup_model_data_feature_ranges(self, sample_train_data, sample_test_data):
        """Test that features are within expected ranges."""
        (
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            x_percentage_test,
            x_delta_wins_test,
            coords,
        ) = setup_model_data(sample_train_data, sample_test_data)

        # Percentages should be between 0 and 1
        assert np.all(x_percentage_train >= 0) and np.all(x_percentage_train <= 1)
        assert np.all(x_percentage_test >= 0) and np.all(x_percentage_test <= 1)


class TestFitBayesianModel:
    """Tests for fit_bayesian_model function."""

    def test_fit_bayesian_model_returns_idata(self, sample_train_data, sample_test_data):
        """Test that fit_bayesian_model returns an idata object."""
        (
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            x_percentage_test,
            x_delta_wins_test,
            coords,
        ) = setup_model_data(sample_train_data, sample_test_data)

        # Use a small number of samples for faster testing
        idata = fit_bayesian_model(
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            coords,
            random_seed=42,
        )

        # Check that idata has expected structure
        assert hasattr(idata, "posterior")
        assert "a" in idata.posterior.data_vars
        assert "bp" in idata.posterior.data_vars
        assert "bw" in idata.posterior.data_vars

    def test_fit_bayesian_model_reproducible(self, sample_train_data, sample_test_data):
        """Test that same random seed produces similar results."""
        (
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            x_percentage_test,
            x_delta_wins_test,
            coords,
        ) = setup_model_data(sample_train_data, sample_test_data)

        idata1 = fit_bayesian_model(
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            coords,
            random_seed=42,
        )
        idata2 = fit_bayesian_model(
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            coords,
            random_seed=42,
        )

        # Extract intercept from both runs
        a1 = idata1.posterior["a"].values.mean()
        a2 = idata2.posterior["a"].values.mean()

        # Should be very close
        assert np.isclose(a1, a2, atol=0.5)


class TestExtractParameters:
    """Tests for extract_parameters function."""

    def test_extract_parameters_returns_three_values(self, sample_train_data, sample_test_data):
        """Test that extract_parameters returns three values."""
        (
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            x_percentage_test,
            x_delta_wins_test,
            coords,
        ) = setup_model_data(sample_train_data, sample_test_data)

        idata = fit_bayesian_model(
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            coords,
            random_seed=42,
        )

        a_param, beta_p, beta_w = extract_parameters(idata)

        assert isinstance(a_param, (float, np.floating))
        assert isinstance(beta_p, np.ndarray)
        assert isinstance(beta_w, np.ndarray)

    def test_extract_parameters_beta_shapes(self, sample_train_data, sample_test_data):
        """Test that beta parameters have correct shapes."""
        (
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            x_percentage_test,
            x_delta_wins_test,
            coords,
        ) = setup_model_data(sample_train_data, sample_test_data)

        idata = fit_bayesian_model(
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            coords,
            random_seed=42,
        )

        a_param, beta_p, beta_w = extract_parameters(idata)

        assert beta_p.shape == (2,)  # Two percentage coefficients
        assert beta_w.shape == (2,)  # Two delta wins coefficients

    def test_extract_parameters_values_are_numeric(self, sample_train_data, sample_test_data):
        """Test that extracted parameters are numeric."""
        (
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            x_percentage_test,
            x_delta_wins_test,
            coords,
        ) = setup_model_data(sample_train_data, sample_test_data)

        idata = fit_bayesian_model(
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            coords,
            random_seed=42,
        )

        a_param, beta_p, beta_w = extract_parameters(idata)

        assert not np.isnan(a_param)
        assert not np.any(np.isnan(beta_p))
        assert not np.any(np.isnan(beta_w))
