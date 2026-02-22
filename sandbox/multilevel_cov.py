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
import arviz as az
import pytensor.tensor as pt

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

home_games_played, home_games_idx = train_data.to_pandas().home_games_played.factorize()
away_games_played, away_games_idx = train_data.to_pandas().away_games_played.factorize()

y_train = train_data[["home_win"]].to_numpy()[:,0]
x_delta_wins_train = train_data[["prev_home_delta_wins","prev_home_delta_loss"]].to_numpy()

coords = {
    "delta_wins_coeffs": ["prev_home_delta_wins", "prev_home_delta_loss"],
    "home_games_played": home_games_idx,
    "away_games_played": away_games_idx,
    "param": ["intercept", "home_games_slope", "away_games_slope"]
    }

home_percentage_train = train_data["home_team_prev_percentage"].to_numpy()
away_percentage_train = train_data["away_team_prev_percentage"].to_numpy()

# Add covariance between home and away effects
# with pm.Model(coords=coords) as model:
#     home_idx = pm.Data("home_idx", home_games_played, dims="obs_id")
#     away_idx = pm.Data("away_idx", away_games_played, dims="obs_id")
    
#     # data containers
#     home_perc = pm.Data("home_perc", home_percentage_train)
#     away_perc = pm.Data("away_perc", away_percentage_train)
#     X_delta_wins = pm.Data("X_delta_wins", x_delta_wins_train)
#     y = pm.Data("y", y_train)
#     bw = pm.Normal("bw", mu=0, sigma=1, dims="delta_wins_coeffs")
    
#     # Covariance priors
#     sd_dist = pm.HalfCauchy.dist(beta=2)
#     chol_h, corr_h, stds_h  = pm.LKJCholeskyCov('pchol_home', eta=4, n=3, sd_dist=sd_dist)
#     chol_a, corr_a, stds_a  = pm.LKJCholeskyCov('pchol_away', eta=4, n=3, sd_dist=sd_dist)
    
#     # priors for average intercept and slopes:
#     mu_alpha_beta = pm.Normal("mu_alpha_beta", mu=0.0, sigma=5.0, shape=3, dims="param")
    
#     # population of home effects:
#     zh = pm.Normal("zh", 0.0, 1.0, shape=(3, len(home_games_idx)), dims=("param", "home_games_played"))
#     alpha_beta_home = pm.Deterministic(
#         "alpha_beta_home", pt.dot(chol_h, zh).T, dims=("home_games_played", "param")
#     )
#     # population of away effects:
#     za = pm.Normal("za", 0.0, 1.0, shape=(3, len(away_games_idx)), dims=("param", "away_games_played"))
#     alpha_beta_away = pm.Deterministic(
#         "alpha_beta_away", pt.dot(chol_a, za).T, dims=("away_games_played", "param")
#     )
    
#     mu = (mu_alpha_beta[0] +
#          alpha_beta_home[home_idx, 0] + alpha_beta_away[away_idx, 0] +
#          (mu_alpha_beta[1] + alpha_beta_home[home_idx, 1] + alpha_beta_away[away_idx, 1]) * home_perc +
#          (mu_alpha_beta[2] + alpha_beta_home[home_idx, 2] + alpha_beta_away[away_idx, 2]) * away_perc +
#          pm.math.dot(X_delta_wins, bw))

#     # link function
#     p = pm.Deterministic("p", pm.math.invlogit(mu))
#     # likelihood
#     pm.Bernoulli("obs", p=p, observed=y)
#     # fit the model
#     idata = pm.sample(random_seed=123)

coords = {
    "delta_wins_coeffs": ["prev_home_delta_wins", "prev_home_delta_loss"],
    "home_games_played": home_games_idx,
    "away_games_played": away_games_idx,
    "param_int": ["intercept", "home_games_slope", "away_games_slope"],
    "param_h": ["intercept", "home_games_slope"],
    "param_a": ["intercept", "away_games_slope"]
    }

# Remove random effect so it's only home and home and away and away
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
    chol_h, corr_h, stds_h  = pm.LKJCholeskyCov('pchol_home', eta=4, n=2, sd_dist=sd_dist)
    chol_a, corr_a, stds_a  = pm.LKJCholeskyCov('pchol_away', eta=4, n=2, sd_dist=sd_dist)
    # priors for average intercept and slopes:
    mu_alpha_beta = pm.Normal("mu_alpha_beta", mu=0.0, sigma=5.0, shape=3, dims="param_int")
    # population of home effects:
    zh = pm.Normal("zh", 0.0, 1.0, shape=(2, len(home_games_idx)), dims=("param_h", "home_games_played"))
    alpha_beta_home = pm.Deterministic(
        "alpha_beta_home", pt.dot(chol_h, zh).T, dims=("home_games_played", "param_h")
    )
    # population of away effects:
    za = pm.Normal("za", 0.0, 1.0, shape=(2, len(away_games_idx)), dims=("param_a", "away_games_played"))
    alpha_beta_away = pm.Deterministic(
        "alpha_beta_away", pt.dot(chol_a, za).T, dims=("away_games_played", "param_a")
    )
    mu = (mu_alpha_beta[0] +
         alpha_beta_home[home_idx, 0] + alpha_beta_away[away_idx, 0] +
         (mu_alpha_beta[1] + alpha_beta_home[home_idx, 1]) * home_perc +
         (mu_alpha_beta[2] + alpha_beta_away[away_idx, 1]) * away_perc +
         pm.math.dot(X_delta_wins, bw))
    # link function
    p = pm.Deterministic("p", pm.math.invlogit(mu))
    # likelihood
    pm.Bernoulli("obs", p=p, observed=y)
    # fit the model
    idata = pm.sample(random_seed=123, tune=3000, target_accept=0.95)

# az.summary(idata)

# for var in rhat_ds.data_vars:
#     vals = rhat_ds[var].values
#     if np.any(~np.isfinite(vals)):
#         bad[var] = np.unique(vals[~np.isfinite(vals)])

# import arviz as az
# import numpy as np

# rhat = az.rhat(idata)
# corr_rhat = rhat["pchol_home_corr"].values  # shape (chains, draws, n, n) flattened by az.rhat
# # get the matrix R-hat (n x n)
# # For pchol_home_corr in rhat Dataset, you can inspect:
# print(corr_rhat)  
# # But simpler: check which entries are non-finite
# print(np.where(~np.isfinite(corr_rhat)))

# print(idata.sample_stats["diverging"].any())
# print(idata.sample_stats["tree_depth"].max())

# az.summary(idata, var_names=["pchol_home_stds","pchol_away_stds","bw","mu_alpha_beta"])
# az.plot_trace(idata, var_names=["stds_h","stds_a","bw"])

# # show posterior variables
# print(sorted(list(idata.posterior.data_vars)))
# # and show sample_stats keys too
# print(sorted(list(idata.sample_stats.data_vars)))