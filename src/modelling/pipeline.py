"""Pipeline implementation for the modelling package.

This module provides a `run_pipeline` function that performs the data
preparation, model training and prediction generation. It can run either
a standard Bayesian logistic regression or a multilevel logistic regression
based on the model_type configuration.
"""

from src.modelling.data_preparation import (
    load_data,
    process_results,
    calculate_win_loss_records,
    prepare_main_features,
    prepare_feature_dataframe,
    add_games_played_features,
    split_train_test,
    prepare_test_data,
)
from src.modelling.model import (
    setup_model_data,
    fit_bayesian_model,
    extract_parameters,
    setup_multilevel_model_data,
    fit_multilevel_model,
)
from src.modelling.predictions import (
    generate_predictions,
    generate_multilevel_predictions,
)
from src.modelling.config import DATA_CONFIG
from typing import Tuple, List
import polars as pl


def run_pipeline() -> Tuple[pl.DataFrame, List[tuple]]:
    """Run modelling pipeline with configurable model type.

    Performs data preparation and fits either a Bayesian logistic regression
    or a multilevel logistic regression based on DATA_CONFIG["model_type"].
    
    Prints progress messages as each stage completes.

    Returns:
        test_data: Polars DataFrame of fixture enriched with features
        predictions: List of tuples (prob_home, score_home, score_away)
    """
    model_type = DATA_CONFIG.get("model_type", "bayesian").lower()
    
    if model_type not in ["bayesian", "multilevel"]:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'bayesian' or 'multilevel'.")
    
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

    print("Adding games-played features...")
    feature_df = add_games_played_features(results_df, feature_df)

    print("Splitting train/test data...")
    train_data, test_season_data = split_train_test(feature_df, DATA_CONFIG["predict_round"])
    
    print("Preparing test data...")
    test_data = prepare_test_data(fixture, test_season_data, DATA_CONFIG["predict_round"])

    if model_type == "bayesian":
        print("Setting up Bayesian model data...")
        (
            x_percentage_train,
            x_delta_wins_train,
            y_train,
            x_percentage_test,
            x_delta_wins_test,
            coords,
        ) = setup_model_data(train_data, test_data)

        print("Fitting Bayesian logistic regression model (this may take a minute)...")
        idata = fit_bayesian_model(x_percentage_train, x_delta_wins_train, y_train, coords)
        
        print("Extracting parameters...")
        a_param, beta_p, beta_w = extract_parameters(idata)

        print("Generating predictions...")
        predictions = generate_predictions(a_param, beta_p, beta_w, x_percentage_test, x_delta_wins_test)
    
    else:  # multilevel
        print("Setting up multilevel model data...")
        home_idx, away_idx, home_perc, away_perc, x_delta_wins, y, coords = setup_multilevel_model_data(train_data)

        print("Fitting multilevel logistic regression model (this may take a few minutes)...")
        idata = fit_multilevel_model(home_idx, away_idx, home_perc, away_perc, x_delta_wins, y, coords)

        print("Generating multilevel predictions...")
        # Prepare test data indices by factorizing test games-played
        test_home_idx = test_data.to_pandas().home_games_played.values
        test_away_idx = test_data.to_pandas().away_games_played.values
        test_home_perc = test_data["home_team_prev_percentage"].to_numpy()
        test_away_perc = test_data["away_team_prev_percentage"].to_numpy()
        test_x_delta = test_data[["prev_home_delta_wins", "prev_home_delta_loss"]].to_numpy()

        predictions = generate_multilevel_predictions(
            idata, test_home_idx, test_away_idx, test_home_perc, test_away_perc, test_x_delta
        )

    return test_data, predictions
