source("R/data_scrape.r")

config <- yaml::read_yaml("configs/data_config.yaml")[["raw_ladder"]]

year_vector = config[["year_lb"]]:config[["year_ub"]]
round_vector = config[["round_lb"]]:config[["round_ub"]]
data_output_loc = config[["ladder_data_loc"]]

available_cores = parallel::detectCores()
future::plan(future::multisession, workers = available_cores)

if(!dir.exists(data_output_loc)){
    dir.create(data_output_loc, recursive = TRUE)
}

years = sort(rep(year_vector, length(round_vector)))
rounds = rep(round_vector, length(year_vector))

future.apply::future_Map(function(x, y) 
                         tryCatch(scrape_ladder(season = x, round = y, path = data_output_loc), 
                         error = function(e) NULL),
                         x = years, y = rounds)
 
