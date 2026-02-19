"""Bayesian model fitting and parameter extraction."""

import numpy as np
import pymc as pm
import arviz as az
import polars as pl
from typing import Tuple, Dict


def setup_model_data(
    train_data: pl.DataFrame, test_data: pl.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, list]]:
    """Extract feature arrays and coordinates for the model.
    
    Args:
        train_data: Training feature DataFrame
        test_data: Test feature DataFrame
        
    Returns:
        Tuple of (x_percentage_train, x_delta_wins_train, y_train, 
                  x_percentage_test, x_delta_wins_test, coords)
    """
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
    """Fit Bayesian logistic regression model.
    
    Args:
        x_percentage_train: Training data for percentage features
        x_delta_wins_train: Training data for win delta features
        y_train: Training target (home win indicator)
        coords: Coordinate dictionary for dimensions
        random_seed: Random seed for reproducibility
        
    Returns:
        ArviZ InferenceData object with posterior samples
    """
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
    """Extract posterior means for model parameters.
    
    Args:
        idata: ArviZ InferenceData object from model fitting
        
    Returns:
        Tuple of (intercept, beta_percentage, beta_delta_wins)
    """
    a_param = az.summary(idata, var_names=["a"])["mean"]["a"]
    beta_p = np.array(az.summary(idata, var_names=["bp"])["mean"])
    beta_w = np.array(az.summary(idata, var_names=["bw"])["mean"])

    return a_param, beta_p, beta_w
