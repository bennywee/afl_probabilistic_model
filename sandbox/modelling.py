import polars as pl
import pymc as pm
import arviz as az

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

main_features = (
  ladder.select(pl.all().sort_by(['Season','Round.Number']).over('Team').sort_by(['Team']))
        # .with_columns(Percentage = pl.col("Percentage") * 100)
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

predict_round = (2024, 5)
test_year = (pl.col("Season") == predict_round[0]) 
test_round =(pl.col("Round.Number") == predict_round[1])

train_data = feature_df.filter(~(test_year & test_round)) 
test_data = feature_df.filter(test_year & test_round)

x_train = train_data[["home_team_prev_percentage", "away_team_prev_percentage"]].to_numpy()
y_train = train_data[["home_win"]].to_numpy()[:,0]
x_test = test_data[["home_team_prev_percentage", "away_team_prev_percentage"]].to_numpy()
y_test = test_data[["home_win"]].to_numpy()[:,0]

coords = {"coeffs": ["home_team_prev_percentage", "away_team_prev_percentage"]}

# x_train=feature_df[["home_team_prev_percentage", "away_team_prev_percentage"]].to_numpy()
# y_train = feature_df[["home_win"]].to_numpy()[:,0]

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

with model:
    pm.set_data({"X": x_test, "y": y_test})
    idata.extend(pm.sample_posterior_predictive(idata))

p_test_pred = idata.posterior_predictive["obs"].mean(dim=["chain", "draw"])
p_test_pred.to_numpy()

