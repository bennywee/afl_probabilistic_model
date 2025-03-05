import polars as pl
import arviz as az
import pymc as pm

ladder_path = "data/prod/raw/ladder"
results_path = "data/prod/raw/results"

ladder = pl.scan_parquet(ladder_path).collect()
results = pl.scan_parquet(results_path).collect()

results_df = (results.with_columns(
        pl.when(pl.col("Home.Points") > pl.col("Away.Points"))
          .then(1)
          .otherwise(0)
          .alias("home_win"))
        .filter(~pl.col("Round").is_in(["QF", "EF", "SF", "PF", "GF"]))
        .select("Season", "Round.Number", "home_win", "Home.Team", "Away.Team")
)

main_features= (
  ladder.select(pl.all().sort_by(['Season','Round.Number']).over('Team').sort_by(['Team']))
        .with_columns((pl.col("Percentage").shift().over(["Team", "Season"]).name.prefix("prev_")))
        .with_columns((pl.col("prev_Percentage").fill_null(0)))
        .select("Team", "Season", "Round.Number", "Percentage", "prev_Percentage")
)

feature_df = (results_df.join(main_features,
                left_on = ["Season", "Round.Number", "Home.Team"],
                right_on = ["Season", "Round.Number", "Team"],
                how = "left",
                coalesce=True)
              .rename({"Percentage": "home_team_percentage",
                       "prev_Percentage": "home_team_prev_percentage"})
              .join(main_features,
                left_on = ["Season", "Round.Number", "Away.Team"],
                right_on = ["Season", "Round.Number", "Team"],
                how = "left",
                coalesce=True)
              .rename({"Percentage": "away_team_percentage",
                       "prev_Percentage": "away_team_prev_percentage"})
)


feature_df[["home_team_prev_percentage"]].shape
feature_df[["home_win"]].to_numpy().shape

with pm.Model() as logistic_model:
    x_data = pm.Data("x", feature_df[["home_team_prev_percentage"]])
    y_data = pm.Data("y", feature_df[["home_win"]])

    alpha = pm.Normal("alpha")
    beta = pm.Normal("beta")

    p = pm.Deterministic("p", pm.math.sigmoid(alpha + beta * x_data))

    # Here is were we link the shapes of the inputs (x_data) and the observed varaiable
    # It will be the shape we tell it, rather than the (constant!) shape of y_data
    obs = pm.Bernoulli("obs", p=p, observed=y_data)

    # fit the model
    idata = pm.sample(random_seed=123)

    # Generate a counterfactual dataset using our model
    idata = pm.sample_posterior_predictive(
        idata, extend_inferencedata=True, random_seed=123
    )



t = feature_df[["home_win", "home_team_prev_percentage", "away_team_prev_percentage"]]

y = feature_df[["home_win"]]
features = feature_df[["home_team_prev_percentage", "away_team_prev_percentage"]]

coords={
 "y": "home_win",
 "features": ["home_team_prev_percentage", "away_team_prev_percentage"],
}


with pm.Model(coords=coords) as logistic_model:
    x = pm.Data("x", features, dims=["features"])

    alpha = pm.Normal("alpha")
    betas = pm.Normal("betas", dims="features")

    p = pm.Deterministic("p", pm.math.sigmoid(alpha + beta * x))

    # Here is were we link the shapes of the inputs (x_data) and the observed varaiable
    # It will be the shape we tell it, rather than the (constant!) shape of y_data
    obs = pm.Bernoulli("obs", p=p, dims="y")

    # fit the model
    idata = pm.sample(random_seed=123)

    # Generate a counterfactual dataset using our model
    idata = pm.sample_posterior_predictive(
        idata, extend_inferencedata=True, random_seed=123
    )

feature_df[["home_team_prev_percentage"]].to_numpy()

with pm.Model() as logistic_model:
    alpha = pm.Normal("alpha")
    beta = pm.Normal("beta", shape = 2)

    p = pm.Deterministic("p", pm.math.sigmoid(alpha + beta[0] * feature_df[["home_team_prev_percentage"]].to_numpy() + beta[1] * feature_df[["away_team_prev_percentage"]].to_numpy()))

    # Here is were we link the shapes of the inputs (x_data) and the observed varaiable
    # It will be the shape we tell it, rather than the (constant!) shape of y_data
    obs = pm.Bernoulli("obs", p=p, observed=feature_df[["home_win"]].to_numpy())

    # fit the model
    idata = pm.sample(random_seed=123)

    # Generate a counterfactual dataset using our model
    idata = pm.sample_posterior_predictive(
        idata, extend_inferencedata=True, random_seed=123
    )


# Worked
with pm.Model() as logistic_model:
    alpha = pm.Normal("alpha")
    beta = pm.Normal("beta", shape = 2)

    p = pm.Deterministic("p", pm.math.sigmoid(alpha + beta[0] * feature_df[["home_team_prev_percentage"]].to_numpy() + beta[1] * feature_df[["away_team_prev_percentage"]].to_numpy()))

    # Here is were we link the shapes of the inputs (x_data) and the observed varaiable
    # It will be the shape we tell it, rather than the (constant!) shape of y_data
    obs = pm.Bernoulli("obs", p=p, observed=feature_df[["home_win"]].to_numpy())

    # fit the model
    idata = pm.sample(random_seed=123)

    # Generate a counterfactual dataset using our model
    idata = pm.sample_posterior_predictive(
        idata, extend_inferencedata=True, random_seed=123
    )

# Didn't work
t=feature_df[["home_team_prev_percentage", "away_team_prev_percentage"]].to_numpy()
y = feature_df[["home_win"]].to_numpy()[:,0]

with pm.Model() as logistic_model:
    x_data = pm.Data("x_data", t)
    y_data = pm.Data("y_data", y)

    alpha = pm.Normal("alpha")
    beta = pm.Normal("beta", dims = x_data)

    p = pm.Deterministic("p", pm.math.sigmoid(alpha + beta * x_data))

    # Here is were we link the shapes of the inputs (x_data) and the observed varaiable
    # It will be the shape we tell it, rather than the (constant!) shape of y_data
    obs = pm.Bernoulli("obs", p=p, observed=y_data)

    # fit the model
    idata = pm.sample(random_seed=123)

    # Generate a counterfactual dataset using our model
    idata = pm.sample_posterior_predictive(
        idata, extend_inferencedata=True, random_seed=123
    )

import numpy as np

coords = {
    "obsv": np.arange(t.shape[0]),
    "predictors": ["home_team_prev_percentage", "away_team_prev_percentage"]
}

coords = {
    "obsv": range(t.shape[0]),
    "predictors": ["home_team_prev_percentage", "away_team_prev_percentage"]
}

with pm.Model(coords=coords) as logistic_model:
    x_data = pm.Data("x_data", t, dims = ["obsv", "predictors"])
    y_data = pm.Data("y_data", y)

    alpha = pm.Normal("alpha")
    beta = pm.Normal("beta", dims = "predictors")

    p = pm.Deterministic("p", pm.math.sigmoid(alpha + beta * x_data))

    # Here is were we link the shapes of the inputs (x_data) and the observed varaiable
    # It will be the shape we tell it, rather than the (constant!) shape of y_data
    obs = pm.Bernoulli("obs", p=p, observed=y_data, dims="obsv")

    # fit the model
    idata = pm.sample(random_seed=123)

    # Generate a counterfactual dataset using our model
    idata = pm.sample_posterior_predictive(
        idata, extend_inferencedata=True, random_seed=123
    )



coords = {"coeffs": ["home_team_prev_percentage", "away_team_prev_percentage"]}

x_train=feature_df[["home_team_prev_percentage", "away_team_prev_percentage"]].to_numpy()
y_train = feature_df[["home_win"]].to_numpy()[:,0]

with pm.Model(coords=coords) as model:
    # data containers
    X = pm.MutableData("X", x_train)
    y = pm.MutableData("y", y_train)
    # priors
    a = pm.Normal("a", mu=0, sigma=1)
    b = pm.Normal("b", mu=0, sigma=1, dims="coeffs")
    # linear model
    mu = a + pm.math.dot(X, b)
    # link function
    # p = pm.Deterministic("p", pm.math.invlogit(mu))
    p = pm.Deterministic("p", pm.math.sigmoid(mu))
    # likelihood
    pm.Bernoulli("obs", p=p, observed=y)

    # fit the model
    idata = pm.sample(random_seed=123)


coords = {"coeffs": ["home_team_prev_percentage", "away_team_prev_percentage"]}

x_train=feature_df[["home_team_prev_percentage", "away_team_prev_percentage"]].to_numpy()
y_train = feature_df[["home_win"]].to_numpy()[:,0]


#this works
with pm.Model(coords=coords) as model:
    # data containers
    X = pm.Data("X", x_train)
    y = pm.Data("y", y_train)
    # priors
    a = pm.Normal("a", mu=0, sigma=1)
    b = pm.Normal("b", mu=0, sigma=1, dims="coeffs")
    # linear model
    mu = a + pm.math.dot(X, b)
    # link function
    p = pm.Deterministic("p", pm.math.invlogit(mu))
    # likelihood
    pm.Bernoulli("obs", p=p, observed=y)

    # fit the model
    idata = pm.sample(random_seed=123)

az.plot_trace(idata, var_names="b", compact=False);

coords = {"coeffs": ["home_team_prev_percentage", "away_team_prev_percentage"]}

x_train1=feature_df[["home_team_prev_percentage"]].to_numpy()
x_train2=feature_df[["away_team_prev_percentage"]].to_numpy()
y_train = feature_df[["home_win"]].to_numpy()[:,0]

coords = {"coeffs1": ["home_team_prev_percentage"],
          "coeffs2": ["away_team_prev_percentage"]}

# This works
with pm.Model(coords=coords) as model:
    # data containers
    X1 = pm.Data("X1", x_train1)
    X2 = pm.Data("X2", x_train2)
    y = pm.Data("y", y_train)

    # priors
    a = pm.Normal("a", mu=0, sigma=1)
    b1 = pm.Normal("b1", mu=0, sigma=1, dims="coeffs1")
    b2 = pm.Normal("b2", mu=0, sigma=2, dims="coeffs2")

    # linear model
    mu = a + pm.math.dot(X1, b1) + pm.math.dot(X2, b2)
    # link function
    p = pm.Deterministic("p", pm.math.invlogit(mu))
    # likelihood
    pm.Bernoulli("obs", p=p, observed=y)

    # fit the model
    idata = pm.sample(random_seed=123)

    # data = pm.sample_posterior_predictive(
    #     idata, extend_inferencedata=True, random_seed=123
    # )

pm.model_to_graphviz(model)

az.plot_trace(idata, var_names="b1", compact=False);
az.plot_trace(idata, var_names="b2", compact=False);
az.plot_trace(idata, var_names="p", compact=False);


# Redo