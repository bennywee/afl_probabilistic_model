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
    prepare_test_data
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

import pymc as pm

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


all_games = pl.concat([
    results_df.select("Season", "Round.Number", "Home.Team").rename({"Home.Team": "Team"}),
    results_df.select("Season", "Round.Number", "Away.Team").rename({"Away.Team": "Team"}),
]).sort(["Season", "Team", "Round.Number"])

games_played = (
    all_games
    .with_columns(
        games_count=pl.lit(1)
    )
    .with_columns(
        games_played=pl.col("games_count").cum_sum().over("Season", "Team")
    )
    .select("Season", "Round.Number", "Team", "games_played")
    # Subtract 1 to get games played BEFORE this round
    .with_columns(
        games_played=pl.col("games_played") - 1
    )
)

# Join back to feature_df for home team
feature_df = feature_df.join(
    games_played,
    left_on=["Season", "Round.Number", "Home.Team"],
    right_on=["Season", "Round.Number", "Team"],
    how="left",
).rename({"games_played": "home_games_played"})

# Join for away team  
feature_df = feature_df.join(
    games_played,
    left_on=["Season", "Round.Number", "Away.Team"],
    right_on=["Season", "Round.Number", "Team"],
    how="left",
).rename({"games_played": "away_games_played"})

train_data, test_season_data = split_train_test(feature_df, DATA_CONFIG["predict_round"])
test_data = prepare_test_data(fixture, test_season_data, DATA_CONFIG["predict_round"])

feature_names = [
    "home_team_prev_percentage", 
    "away_team_prev_percentage",
    "prev_home_delta_wins",
    "prev_home_delta_loss",
    "home_games_played",
    "away_games_played"
]

# x_train = train_data[feature_names].to_numpy()
# y_train = train_data[["home_win"]].to_numpy()[:,0]

home_games_played, home_games_idx = train_data.to_pandas().home_games_played.factorize()
away_games_played, away_games_idx = train_data.to_pandas().away_games_played.factorize()

y_train = train_data[["home_win"]].to_numpy()[:,0]
x_percentage_train = train_data["home_team_prev_percentage"].to_numpy()
x_delta_wins_train = train_data[["prev_home_delta_wins","prev_home_delta_loss"]].to_numpy()

coords = {
    "perc_coeffs": ["home_team_prev_percentage"],
    "delta_wins_coeffs" :["prev_home_delta_wins","prev_home_delta_loss"],
    "home_games_played": home_games_idx
    }


with pm.Model(coords=coords) as model:
    home_idx = pm.Data("home_idx", home_games_played, dims="obs_id")
    # data containers
    X_perc = pm.Data("X_perc", x_percentage_train)
    X_delta_wins = pm.Data("X_delta_wins", x_delta_wins_train)
    y = pm.Data("y", y_train)
    bw = pm.Normal("bw", mu=0, sigma=1, dims="delta_wins_coeffs")
    # NC random intercepts
    mu_a = pm.Normal("mu_a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 5)
    z_a = pm.Normal("z_a", mu=0, sigma=1, dims="home_games_played")
    alpha = pm.Deterministic("alpha", mu_a + z_a * sigma_a, dims="home_games_played")
    # NC random slopes
    mu_b = pm.Normal("mu_b", mu=0.0, sigma=10.0)
    sigma_b = pm.Exponential("sigma_b", 5)
    z_b = pm.Normal("z_b", mu=0, sigma=1, dims="home_games_played")
    beta = pm.Deterministic("beta", mu_b + z_b * sigma_b, dims="home_games_played")
    # Expected value
    mu = alpha[home_games_played] + beta[home_games_played] * X_perc + pm.math.dot(X_delta_wins, bw)
    # link function
    p = pm.Deterministic("p", pm.math.invlogit(mu))
    # likelihood
    pm.Bernoulli("obs", p=p, observed=y)
    # fit the model
    idata = pm.sample(random_seed=123)

coords = {
    "delta_wins_coeffs" :["prev_home_delta_wins","prev_home_delta_loss"],
    "home_games_played": home_games_idx,
    "away_games_played": away_games_idx
    }

home_percentage_train = train_data["home_team_prev_percentage"].to_numpy()
away_percentage_train = train_data["home_team_prev_percentage"].to_numpy()


# Add away int
with pm.Model(coords=coords) as model:
    home_idx = pm.Data("home_idx", home_games_played, dims="obs_id")
    away_idx = pm.Data("away_idx", away_games_played, dims="obs_id")
    # data containers
    home_perc = pm.Data("home_perc", home_percentage_train)
    away_perc = pm.Data("away_perc", away_percentage_train)
    X_delta_wins = pm.Data("X_delta_wins", x_delta_wins_train)
    y = pm.Data("y", y_train)
    bw = pm.Normal("bw", mu=0, sigma=1, dims="delta_wins_coeffs")
    # NC random intercepts
    mu_h = pm.Normal("mu_h", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 5)
    z_a = pm.Normal("z_a", mu=0, sigma=1, dims="home_games_played")
    sigma_h = pm.Exponential("sigma_h", 5)
    z_h = pm.Normal("z_h", mu=0, sigma=1, dims="home_games_played") 
    alpha = pm.Deterministic("alpha", mu_h + z_h * sigma_h, dims=["home_games_played"])
    alpha_a = pm.Deterministic("alpha_a", z_a * sigma_a, dims=["away_games_played"])
    # NC random slopes
    mu_b = pm.Normal("mu_b", mu=0.0, sigma=10.0)
    sigma_b = pm.Exponential("sigma_b", 5)
    z_b = pm.Normal("z_b", mu=0, sigma=1, dims="home_games_played")
    beta_home = pm.Deterministic("beta_home", mu_b + z_b * sigma_b, dims="home_games_played")
    # Expected value
    mu = alpha[home_games_played] + alpha_a[away_games_played] + beta_home[home_games_played] * home_perc + pm.math.dot(X_delta_wins, bw)
    # link function
    p = pm.Deterministic("p", pm.math.invlogit(mu))
    # likelihood
    pm.Bernoulli("obs", p=p, observed=y)
    # fit the model
    idata = pm.sample(random_seed=123)

# Add away slope
with pm.Model(coords=coords) as model:
    home_idx = pm.Data("home_idx", home_games_played, dims="obs_id")
    away_idx = pm.Data("away_idx", away_games_played, dims="obs_id")
    # data containers
    home_perc = pm.Data("home_perc", home_percentage_train)
    away_perc = pm.Data("away_perc", away_percentage_train)
    X_delta_wins = pm.Data("X_delta_wins", x_delta_wins_train)
    y = pm.Data("y", y_train)
    bw = pm.Normal("bw", mu=0, sigma=1, dims="delta_wins_coeffs")
    # NC random intercepts
    mu_h = pm.Normal("mu_h", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 5)
    z_a = pm.Normal("z_a", mu=0, sigma=1, dims="home_games_played")
    sigma_h = pm.Exponential("sigma_h", 5)
    z_h = pm.Normal("z_h", mu=0, sigma=1, dims="home_games_played") 
    alpha = pm.Deterministic("alpha", mu_h + z_h * sigma_h, dims=["home_games_played"])
    alpha_a = pm.Deterministic("alpha_a", z_a * sigma_a, dims=["away_games_played"])
    # NC random slopes
    mu_b = pm.Normal("mu_b", mu=0.0, sigma=10.0)
    sigma_b = pm.Exponential("sigma_b", 5)
    z_b = pm.Normal("z_b", mu=0, sigma=1, dims="home_games_played")
    beta_home = pm.Deterministic("beta_home", mu_b + z_b * sigma_b, dims="home_games_played")
    sigma_as = pm.Exponential("sigma_as", 5)
    z_as = pm.Normal("z_as", mu=0, sigma=1, dims="away_games_played")
    beta_away = pm.Deterministic("beta_away", z_as * sigma_as, dims="away_games_played")
    # Expected value
    mu = alpha[home_games_played] + alpha_a[away_games_played] + beta_home[home_games_played] * home_perc + beta_away[away_games_played] * away_perc + pm.math.dot(X_delta_wins, bw)
    # link function
    p = pm.Deterministic("p", pm.math.invlogit(mu))
    # likelihood
    pm.Bernoulli("obs", p=p, observed=y)
    # fit the model
    idata1 = pm.sample(random_seed=123)

    # Add covariance between home and away effects
with pm.Model(coords=coords) as model:
    home_idx = pm.Data("home_idx", home_games_played, dims="obs_id")
    away_idx = pm.Data("away_idx", away_games_played, dims="obs_id")
    # data containers
    home_perc = pm.Data("home_perc", home_percentage_train)
    away_perc = pm.Data("away_perc", away_percentage_train)
    X_delta_wins = pm.Data("X_delta_wins", x_delta_wins_train)
    y = pm.Data("y", y_train)
    bw = pm.Normal("bw", mu=0, sigma=1, dims="delta_wins_coeffs")
    # Covariance priors
    sd_dist = pm.HalfCauchy.dist(beta=2)
    pchol1 = pm.LKJCholeskyCov('pchol_home', eta=4, n=2, sd_dist=sd_dist)
    pchol2 = pm.LKJCholeskyCov('pchol_away', eta=4, n=2, sd_dist=sd_dist)
    chol1 = pm.expand_packed_triangular(2, pchol1, lower=True)
    chol2 = pm.expand_packed_triangular(2, pchol2, lower=True)
    Intercept = pm.Normal('intercept', 0., 1., shape=3)
    beta_home = pm.MvNormal('beta_home', mu=0., chol=chol1, shape=(home_games_idx, 2))
    beta_away = pm.MvNormal('beta_away', mu=0., chol=chol2, shape=(away_games_idx, 2))
    A = Intercept[0] + beta_home[home_games_played, 0] + beta_away[away_games_played, 0]
    BH = Intercept[1] +beta_home[home_games_played, 1]
    BA = Intercept[2] +beta_away[beta_away, 1]
    mu = A + BH * home_perc + BA * away_perc + pm.math.dot(X_delta_wins, bw)
    # link function
    p = pm.Deterministic("p", pm.math.invlogit(mu))
    # likelihood
    pm.Bernoulli("obs", p=p, observed=y)
    # fit the model
    idata = pm.sample(random_seed=123)
