"""Prediction generation utilities."""

import numpy as np
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
        List of tuples (prob_home_win, log_odds_home, log_odds_away)
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
