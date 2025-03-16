source("R/data_scrape.r")

config <- yaml::read_yaml("configs/data_config.yaml")
tables <- config[["raw_data_scrape"]][["tables"]]
git_branch <- system("git rev-parse --abbrev-ref HEAD", intern = TRUE)
env <- if(git_branch == "main") "prod" else "dev"

available_cores = parallel::detectCores()-1
future::plan(future::multisession, workers = available_cores)

scrape_table_data <- function(config, table_name) {
    config_tables <-  config[["raw_data_scrape"]][[table_name]]
    years <- config_tables[["year_lb"]]:config_tables[["year_ub"]]
    rounds <- config_tables[["round_lb"]]:config_tables[["round_ub"]]
    data_output_loc = paste0("data/", env, "/", config_tables[["data_loc"]])

    if(!dir.exists(data_output_loc)){
        dir.create(data_output_loc, recursive = TRUE)
    }

    Map(function(season_vector) 
        write_raw_data(table_type = table_name,
                       season = season_vector, 
                       round = rounds, 
                       output_path = data_output_loc),
        season_vector = years
    )
}

Map(function(t) 
    scrape_table_data(config = config, table_name = t),
    t = tables)

# Get 2025 fixture data
fixture_loc <- paste0("data/", env, "/raw/fixture")

if(!dir.exists(fixture_loc)){
    dir.create(fixture_loc, recursive = TRUE)
}

fitzRoy::fetch_fixture_footywire(
  season = 2025
) |> 
arrow::write_dataset(format = "parquet",
                     path = fixture_loc)
