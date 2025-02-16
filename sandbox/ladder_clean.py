import polars as pl

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

        