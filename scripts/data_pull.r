source("R/data_scrape.r")

config <- yaml::read_yaml("configs/data_config.yaml")[["raw_ladder"]]
git_branch <- system("git rev-parse --abbrev-ref HEAD", intern = TRUE)
env <- if(git_branch == "main") "prod" else "dev"

years = config[["year_lb"]]:config[["year_ub"]]
rounds = config[["round_lb"]]:config[["round_ub"]]
data_output_loc = paste0("data/", env, "/", config[["ladder_data_loc"]])

available_cores = parallel::detectCores()-1
future::plan(future::multisession, workers = available_cores)

if(!dir.exists(data_output_loc)){
    dir.create(data_output_loc, recursive = TRUE)
}

Map(function(season_vector) 
    tryCatch(scrape_write_ladder(season = season_vector, 
                                 round = rounds, 
                                 output_path = data_output_loc), 
    error = function(e){print(e); return(NULL)}), 
    season_vector = years
)

