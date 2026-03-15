"""Bayesian model fitting and parameter extraction."""

import numpy as np
import pymc as pm
import arviz as az
import polars as pl
import pytensor.tensor as pt
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


def setup_multilevel_model_data(
    train_data: pl.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Extract and prepare data for multilevel logistic regression model.
    
    Factorizes home_games_played and away_games_played into integer indices,
    extracts feature arrays, and creates coordinate mappings.
    
    Args:
        train_data: Training DataFrame with features including home_games_played, 
                   away_games_played, home_team_prev_percentage, away_team_prev_percentage,
                   prev_home_delta_wins, prev_home_delta_loss, and home_win columns
        
    Returns:
        Tuple of (home_idx, away_idx, home_perc, away_perc, x_delta_wins, y, coords)
    """
    # Factorize games-played indices
    home_games_played, home_games_idx = train_data.to_pandas().home_prev_games_played.factorize()
    away_games_played, away_games_idx = train_data.to_pandas().away_prev_games_played.factorize()
    
    # Extract feature arrays
    home_perc = train_data["home_team_prev_percentage"].to_numpy()
    away_perc = train_data["away_team_prev_percentage"].to_numpy()
    x_delta_wins = train_data[["prev_home_delta_wins", "prev_home_delta_loss"]].to_numpy()
    y = train_data[["home_win"]].to_numpy()[:, 0]
    
    # Create coordinates for the model
    coords = {
        "delta_wins_coeffs": ["prev_home_delta_wins", "prev_home_delta_loss"],
        "home_games_played": home_games_idx,
        "away_games_played": away_games_idx,
        "param_int": ["intercept", "home_games_slope", "away_games_slope"],
        "param_h": ["intercept", "home_games_slope"],
        "param_a": ["intercept", "away_games_slope"],
    }
    
    return home_games_played, away_games_played, home_perc, away_perc, x_delta_wins, y, coords


def fit_multilevel_model(
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_perc: np.ndarray,
    away_perc: np.ndarray,
    x_delta_wins: np.ndarray,
    y: np.ndarray,
    coords: Dict,
    random_seed: int = 123,
    tune: int = 3000,
    target_accept: float = 0.95,
) -> pm.model.Model:
    """Fit multilevel logistic regression model with LKJ priors on correlations.
    
    Uses hierarchical random intercepts and slopes for home and away games groups,
    with LKJ Cholesky covariance priors on both group effects.
    
    Args:
        home_idx: Integer array indexing home-games group for each observation
        away_idx: Integer array indexing away-games group for each observation
        home_perc: Array of home team percentage features
        away_perc: Array of away team percentage features
        x_delta_wins: Array shape (n_obs, 2) of delta-wins features
        y: Binary target (home win indicator)
        coords: Coordinate dictionary from setup_multilevel_model_data
        random_seed: Random seed for reproducibility
        tune: Number of tuning steps
        target_accept: Target acceptance rate for HMC
        
    Returns:
        ArviZ InferenceData object with posterior samples
    """
    with pm.Model(coords=coords) as model:
        # Data containers
        home_idx_data = pm.Data("home_idx", home_idx, dims="obs_id")
        away_idx_data = pm.Data("away_idx", away_idx, dims="obs_id")
        home_perc_data = pm.Data("home_perc", home_perc)
        away_perc_data = pm.Data("away_perc", away_perc)
        X_delta_wins = pm.Data("X_delta_wins", x_delta_wins)
        y_data = pm.Data("y", y)
        
        # Delta-wins coefficients (fixed effects)
        bw = pm.Normal("bw", mu=0, sigma=1, dims="delta_wins_coeffs")
        
        # Covariance priors for random effects
        sd_dist = pm.HalfCauchy.dist(beta=2)
        chol_h, corr_h, stds_h = pm.LKJCholeskyCov("pchol_home", eta=4, n=2, sd_dist=sd_dist)
        chol_a, corr_a, stds_a = pm.LKJCholeskyCov("pchol_away", eta=4, n=2, sd_dist=sd_dist)
        
        # Population-level intercept and slopes
        mu_alpha_beta = pm.Normal("mu_alpha_beta", mu=0.0, sigma=5.0, shape=3, dims="param_int")
        
        # Standardized random effects per group
        zh = pm.Normal("zh", 0.0, 1.0, shape=(2, len(coords["home_games_played"])), dims=("param_h", "home_games_played"))
        za = pm.Normal("za", 0.0, 1.0, shape=(2, len(coords["away_games_played"])), dims=("param_a", "away_games_played"))
        
        # Transform to correlated random effects via Cholesky decomposition
        alpha_beta_home = pm.Deterministic(
            "alpha_beta_home", pt.dot(chol_h, zh).T, dims=("home_games_played", "param_h")
        )
        alpha_beta_away = pm.Deterministic(
            "alpha_beta_away", pt.dot(chol_a, za).T, dims=("away_games_played", "param_a")
        )
        
        # Linear predictor
        mu = (
            mu_alpha_beta[0]
            + alpha_beta_home[home_idx_data, 0]
            + alpha_beta_away[away_idx_data, 0]
            + (mu_alpha_beta[1] + alpha_beta_home[home_idx_data, 1]) * home_perc_data
            + (mu_alpha_beta[2] + alpha_beta_away[away_idx_data, 1]) * away_perc_data
            + pm.math.dot(X_delta_wins, bw)
        )
        
        # Link function
        p = pm.Deterministic("p", pm.math.invlogit(mu))
        
        # Likelihood
        pm.Bernoulli("obs", p=p, observed=y_data)
        
        # Fit the model
        idata = pm.sample(random_seed=random_seed, tune=tune, target_accept=target_accept, progressbar=True)
    
    return idata
