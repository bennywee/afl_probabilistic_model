import polars as pl
import pymc as pm
import numpy as np
import arviz as az

ladder_path = "data/dev/raw/ladder"
results_path = "data/dev/raw/results"
fixture_path = "data/prod/raw/fixture"
predict_round = (2025, 4)

ladder = pl.scan_parquet(ladder_path).collect()
results = pl.scan_parquet(results_path).collect()
fixture = pl.scan_parquet(fixture_path).collect()\
            .select("Round", "Home.Team", "Away.Team")\
            .with_columns(Round = pl.col("Round") - 1)

results_df = (results.with_columns(
        pl.when(pl.col("Home.Points") > pl.col("Away.Points"))
          .then(1)
          .otherwise(0)
          .alias("home_win"))
        .filter(~pl.col("Round").is_in(["QF", "EF", "SF", "PF", "GF"]))
        .select("Season", "Round.Number", "home_win", "Home.Team", "Away.Team")
)

home = results_df.drop("Away.Team").rename({"home_win":"win", "Home.Team":"Team"})
away = (results_df.drop("Home.Team")
           .with_columns(win = 1 - pl.col("home_win"))
           .drop("home_win")
           .rename({"Away.Team":"Team"})
           .select("Season", "Round.Number", "win", "Team"))

total_win_loss = (pl.concat([home, away], how = "vertical").sort(["Season", "Round.Number"])
   .with_columns(loss = 1-pl.col("win"),
                 total_wins = pl.col("win").cum_sum().over("Season", "Team"))
   .with_columns(total_loss = pl.col("loss").cum_sum().over("Season", "Team"))
   .select(pl.all().sort_by(['Season','Round.Number']).over('Team').sort_by(['Team']))
   .with_columns((pl.col("total_wins").shift().over(["Team", "Season"]).name.prefix("prev_")))
   .with_columns((pl.col("prev_total_wins").fill_null(0)))
   .with_columns((pl.col("total_loss").shift().over(["Team", "Season"]).name.prefix("prev_")))
   .with_columns((pl.col("prev_total_loss").fill_null(0)))
   .drop(["win", "loss"]))
                 

main_features = (
  ladder.select(pl.all().sort_by(['Season','Round.Number']).over('Team').sort_by(['Team']))
        .with_columns((pl.col("Percentage").shift().over(["Team", "Season"]).name.prefix("prev_")))
        .with_columns((pl.col("prev_Percentage").fill_null(0)))
        .select("Team", "Season", "Round.Number", "Percentage", "prev_Percentage")
        .join(total_win_loss, how = "left", on = ["Team", "Season", "Round.Number"])
)

feature_df = (results_df.join(main_features,
                left_on = ["Season", "Round.Number", "Home.Team"],
                right_on = ["Season", "Round.Number", "Team"],
                how = "left",
                coalesce=True)
              .rename({"Percentage": "home_team_percentage",
                       "prev_Percentage": "home_team_prev_percentage",
                       "total_wins":"home_total_wins",
                       "total_loss":"home_total_loss",
                       "prev_total_wins":"home_prev_total_wins",
                       "prev_total_loss":"home_prev_total_loss"
                       })
              .join(main_features,
                left_on = ["Season", "Round.Number", "Away.Team"],
                right_on = ["Season", "Round.Number", "Team"],
                how = "left",
                coalesce=True)
              .rename({"Percentage": "away_team_percentage",
                       "prev_Percentage": "away_team_prev_percentage",
                       "total_wins":"away_total_wins",
                       "total_loss":"away_total_loss",
                       "prev_total_wins":"away_prev_total_wins",
                       "prev_total_loss":"away_prev_total_loss"})
              .with_columns(prev_home_win_percentage = (pl.col("home_prev_total_wins") / (pl.col("home_prev_total_wins") + pl.col("home_prev_total_loss"))).fill_nan(0),
                            prev_away_win_percentage = (pl.col("away_prev_total_wins") / (pl.col("away_prev_total_wins") + pl.col("away_prev_total_loss"))).fill_nan(0),
                            home_win_percentage = (pl.col("home_total_wins") / (pl.col("home_total_wins") + pl.col("home_total_loss"))).fill_nan(0),
                            away_win_percentage = (pl.col("away_total_wins") / (pl.col("away_total_wins") + pl.col("away_total_loss"))).fill_nan(0),
                            prev_home_delta_wins = pl.col("home_prev_total_wins") - pl.col("away_prev_total_wins"),
                            prev_home_delta_loss = pl.col("home_prev_total_loss") - pl.col("away_prev_total_loss")
                            )
              .with_columns(Round = pl.col("Round.Number") - 1)

)

test_year = (pl.col("Season") == predict_round[0]) 
test_round =(pl.col("Round") == predict_round[1])
train_data = feature_df.filter(~(test_year & test_round)) 

# _test_data = (feature_df.filter((test_year) & (pl.col("Round") == (predict_round[1]-1))))
_test_data = feature_df.filter((test_year))

_test_home = (_test_data.select("Home.Team",
                              "home_team_percentage",
                              "home_total_wins",
                              "home_total_loss",
                              "Round.Number")
                        .rename({"Home.Team": "Team",
                                 "home_team_percentage":"percentage",
                                 "home_total_wins":"total_wins",
                                 "home_total_loss":"total_loss"})
)

_test_away = (_test_data.select("Away.Team",
                              "away_team_percentage",
                              "away_total_wins",
                              "away_total_loss",
                              "Round.Number")
                        .rename({"Away.Team": "Team",
                                 "away_team_percentage":"percentage",
                                 "away_total_wins":"total_wins",
                                 "away_total_loss":"total_loss"})
)

_test_features = pl.concat([_test_home, _test_away], how = "vertical")
test_features = (_test_features.group_by("Team")
              .agg(pl.col("Round.Number").max())
              .join(_test_features, how = "inner", on = ["Team", "Round.Number"])
              .drop("Round.Number"))

test_data = (fixture.filter(pl.col("Round") == predict_round[1])
  .join(test_features, how = "left", left_on = "Home.Team", right_on = "Team")
  .rename({"percentage":"home_team_prev_percentage",
           "total_wins":"home_prev_total_wins",
           "total_loss":"home_prev_total_loss"})
  .join(test_features, how = "left", left_on = "Away.Team", right_on = "Team")
  .rename({"percentage":"away_team_prev_percentage",
           "total_wins":"away_prev_total_wins",
           "total_loss":"away_prev_total_loss"})
  .with_columns(
    prev_home_delta_wins = pl.col("home_prev_total_wins") - pl.col("away_prev_total_wins"),
    prev_home_delta_loss = pl.col("home_prev_total_loss") - pl.col("away_prev_total_loss")
  )
)



# _test_data = (
#   ladder.with_columns(Round = pl.col("Round.Number") - 1)
#         .filter(test_year & (pl.col("Round") == predict_round[1]-1))
#         .select("Team", "Percentage")
# )

# test_data = (fixture.filter(pl.col("Round") == predict_round[1])
#   .join(_test_data, how = "left", left_on = "Home.Team", right_on = "Team")
#   .rename({"Percentage":"home_team_prev_percentage"})
#   .join(_test_data, how = "left", left_on = "Away.Team", right_on = "Team")
#   .rename({"Percentage":"away_team_prev_percentage"}))
feature_names = ["home_team_prev_percentage", "away_team_prev_percentage","prev_home_delta_wins","prev_home_delta_loss"]
x_train = train_data[feature_names].to_numpy()
y_train = train_data[["home_win"]].to_numpy()[:,0]
x_test = test_data[feature_names].to_numpy()

x_percentage_train = train_data["home_team_prev_percentage", "away_team_prev_percentage"].to_numpy()
x_delta_wins_train = train_data["prev_home_delta_wins","prev_home_delta_loss"].to_numpy()

x_percentage_test = test_data["home_team_prev_percentage", "away_team_prev_percentage"].to_numpy()
x_delta_wins_test = test_data["prev_home_delta_wins","prev_home_delta_loss"].to_numpy()

# y_test = test_data[["home_win"]].to_numpy()[:,0]

coords = {"perc_coeffs": ["home_team_prev_percentage", "away_team_prev_percentage"], "delta_wins_coeffs" :["prev_home_delta_wins","prev_home_delta_loss"]}

# x_train=feature_df[["home_team_prev_percentage", "away_team_prev_percentage"]].to_numpy()
# y_train = feature_df[["home_win"]].to_numpy()[:,0]

with pm.Model(coords=coords) as model:
    # data containers
    X_perc = pm.Data("X_perc", x_percentage_train)
    X_delta_wins = pm.Data("X_delta_wins", x_delta_wins_train)
    y = pm.Data("y", y_train)
    
    # priors
    a = pm.Normal("a", mu=0, sigma=0.1)
    bp = pm.Normal("bp", mu=0, sigma=0.05, dims="perc_coeffs")
    bw = pm.Normal("bw", mu=0, sigma=0.05, dims="delta_wins_coeffs")

    # linear model
    mu = a + pm.math.dot(X_perc, bp) + pm.math.dot(X_delta_wins, bw)
    # link function
    p = pm.Deterministic("p", pm.math.invlogit(mu))
    # likelihood
    pm.Bernoulli("obs", p=p, observed=y)

    # fit the model
    idata = pm.sample(random_seed=123)

a_param = az.summary(idata, var_names = ["a"])["mean"]["a"]
beta_p = np.array(az.summary(idata, var_names = ["bp"])["mean"])
beta_w = np.array(az.summary(idata, var_names = ["bw"])["mean"])
p_array = a_param + np.dot(x_percentage_test, beta_p) + np.dot(x_delta_wins_test, beta_w)

def logistic(p):
    return 1/(1+np.e**(-p))

[(logistic(p), 1+np.log2(logistic(p)), 1+np.log2(1-logistic(p)))  for p in p_array]
