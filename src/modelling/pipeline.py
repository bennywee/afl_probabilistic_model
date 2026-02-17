"""Pipeline implementation for the modelling package.

This module provides a `run_pipeline` function that performs the data
preparation, model training and prediction generation. It includes progress
printing so users can see what stage the pipeline is at.
"""

from src.modelling.data_preparation import (
    load_data,
    process_results,
    calculate_win_loss_records,
    prepare_main_features,
    prepare_feature_dataframe,
    split_train_test,
    prepare_test_data,
)
from src.modelling.model import (
    setup_model_data,
    fit_bayesian_model,
    extract_parameters,
)
from src.modelling.predictions import (
    generate_predictions,
)
from src.modelling.config import DATA_CONFIG
from typing import Tuple, List
import polars as pl


def run_pipeline() -> Tuple[pl.DataFrame, List[tuple]]:
    """Run full modelling pipeline and return test data and predictions.

    Prints progress messages as each stage completes.

    Returns:
        test_data: Polars DataFrame of fixture enriched with features
        predictions: List of tuples produced by `generate_predictions`
    """
    print("Loading data...")
    ladder, results, fixture = load_data(
        DATA_CONFIG["ladder_path"],
        DATA_CONFIG["results_path"],
        DATA_CONFIG["fixture_path"],
    )

    print("Processing results...")
    results_df = process_results(results)
    
    print("Calculating win/loss records...")
    total_win_loss = calculate_win_loss_records(results_df)
    
    print("Preparing main features...")
    main_features = prepare_main_features(ladder, total_win_loss)
    
    print("Preparing feature dataframe...")
    feature_df = prepare_feature_dataframe(results_df, main_features)

    print("Splitting train/test data...")
    train_data, test_season_data = split_train_test(feature_df, DATA_CONFIG["predict_round"])
    
    print("Preparing test data...")
    test_data = prepare_test_data(fixture, test_season_data, DATA_CONFIG["predict_round"])

    print("Setting up model data...")
    (
        x_percentage_train,
        x_delta_wins_train,
        y_train,
        x_percentage_test,
        x_delta_wins_test,
        coords,
    ) = setup_model_data(train_data, test_data)

    print("Fitting Bayesian model (this may take a minute)...")
    idata = fit_bayesian_model(x_percentage_train, x_delta_wins_train, y_train, coords)
    
    print("Extracting parameters...")
    a_param, beta_p, beta_w = extract_parameters(idata)

    print("Generating predictions...")
    predictions = generate_predictions(a_param, beta_p, beta_w, x_percentage_test, x_delta_wins_test)

    return test_data, predictions

