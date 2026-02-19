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
