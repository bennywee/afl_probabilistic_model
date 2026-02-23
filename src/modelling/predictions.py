"""Prediction generation and display utilities."""

import numpy as np
import polars as pl
from typing import List, Tuple


def logistic(x: np.ndarray) -> np.ndarray:
    """Apply logistic (sigmoid) function.
    
    Args:
        x: Input array
        
    Returns:
        Logistic function applied element-wise
    """
    return 1 / (1 + np.exp(-x))


def generate_predictions(
    a_param: float,
    beta_p: np.ndarray,
    beta_w: np.ndarray,
    x_percentage_test: np.ndarray,
    x_delta_wins_test: np.ndarray,
) -> List[Tuple[float, float, float]]:
    """Generate predictions for test data.
    
    Args:
        a_param: Intercept parameter
        beta_p: Percentage feature coefficients
        beta_w: Win delta feature coefficients
        x_percentage_test: Test data for percentage features
        x_delta_wins_test: Test data for win delta features
        
    Returns:
        List of tuples (prob_home_win, score_home_wins, score_away_wins)
    """
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


def print_predictions(
    test_data: pl.DataFrame,
    predictions: List[Tuple[float, float, float]],
) -> None:
    """Print predictions in a formatted table.
    
    Args:
        test_data: Polars DataFrame with Home.Team and Away.Team columns
        predictions: List of tuples from generate_predictions
    """
    print("\nPredictions (Probability of Home Win, Score if Home or Away Wins):")
    
    test_data_dict = test_data.select("Home.Team", "Away.Team").to_dicts()
    for i, pred in enumerate(predictions):
        home_team = test_data_dict[i]["Home.Team"]
        away_team = test_data_dict[i]["Away.Team"]
        prob_home, score_home_wins, score_away_wins = pred
        print(
            f"{home_team:12} vs {away_team:12} | "
            f"P(Home)={prob_home:.3f} | "
            f"Score if H or A wins: {score_home_wins:.3f} (H) / {score_away_wins:.3f} (A)"
        )


def extract_multilevel_means(idata) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract posterior means for multilevel model components.

    Returns:
        mu_alpha_beta_mean: shape (3,) population intercept and slopes means
        alpha_beta_home_mean: array shape (n_home_groups, 2) group intercept and home-slope means
        alpha_beta_away_mean: array shape (n_away_groups, 2) group intercept and away-slope means
        bw_mean: array shape (2,) delta-wins coefficients means
    """
    posterior = idata.posterior
    # population-level intercepts/slopes
    mu_alpha_beta_mean = posterior["mu_alpha_beta"].mean(dim=("chain", "draw")).values

    # group-level deterministic arrays (means over posterior)
    alpha_beta_home_mean = posterior["alpha_beta_home"].mean(dim=("chain", "draw")).values
    alpha_beta_away_mean = posterior["alpha_beta_away"].mean(dim=("chain", "draw")).values

    # delta-wins coefficients
    bw_mean = posterior["bw"].mean(dim=("chain", "draw")).values

    return mu_alpha_beta_mean, alpha_beta_home_mean, alpha_beta_away_mean, bw_mean


def generate_multilevel_predictions(
    idata,
    home_idx_test: np.ndarray,
    away_idx_test: np.ndarray,
    home_perc_test: np.ndarray,
    away_perc_test: np.ndarray,
    x_delta_wins_test: np.ndarray,
) -> List[Tuple[float, float, float]]:
    """Generate predictions for test data using posterior means from a multilevel model.

    Args:
        idata: ArviZ InferenceData from the fitted multilevel model
        home_idx_test: integer array indexing home group for each test row
        away_idx_test: integer array indexing away group for each test row
        home_perc_test: array of home percentage feature for test rows
        away_perc_test: array of away percentage feature for test rows
        x_delta_wins_test: array shape (n_obs, 2) for delta-wins features

    Returns:
        List of tuples (prob_home, score_home_if_win, score_away_if_win) per observation
    """
    mu_alpha_beta_mean, alpha_beta_home_mean, alpha_beta_away_mean, bw_mean = (
        extract_multilevel_means(idata)
    )

    # build linear predictor using posterior means
    preds = []
    for i in range(len(home_idx_test)):
        h = int(home_idx_test[i])
        a = int(away_idx_test[i])

        mu = (
            mu_alpha_beta_mean[0]
            + alpha_beta_home_mean[h, 0]
            + alpha_beta_away_mean[a, 0]
            + (mu_alpha_beta_mean[1] + alpha_beta_home_mean[h, 1]) * home_perc_test[i]
            + (mu_alpha_beta_mean[2] + alpha_beta_away_mean[a, 1]) * away_perc_test[i]
            + np.dot(x_delta_wins_test[i], bw_mean)
        )

        p = logistic(mu)
        preds.append((float(p), float(1 + np.log2(p)), float(1 + np.log2(1 - p))))

    return preds


def print_multilevel_predictions(
    test_data: pl.DataFrame,
    predictions: List[Tuple[float, float, float]],
) -> None:
    """Print predictions produced by `generate_multilevel_predictions`.

    Args:
        test_data: Polars DataFrame with at least `Home.Team` and `Away.Team` columns
        predictions: list of prediction tuples
    """
    print("\nMultilevel model predictions:")
    test_data_dict = test_data.select("Home.Team", "Away.Team").to_dicts()
    for i, pred in enumerate(predictions):
        home_team = test_data_dict[i]["Home.Team"]
        away_team = test_data_dict[i]["Away.Team"]
        prob_home, score_home_wins, score_away_wins = pred
        print(
            f"{home_team:12} vs {away_team:12} | P(Home)={prob_home:.3f} | "
            f"Score if H/A wins: {score_home_wins:.3f} / {score_away_wins:.3f}"
        )
