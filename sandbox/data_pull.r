year_vector = 2000:2024
round_vector = 1:25

years = sort(rep(year_vector, length(round_vector)))
rounds = rep(round_vector, length(year_vector))

available_cores = parallel::detectCores()
future::plan(future::multisession, workers = available_cores-1)

data_ls <- future.apply::future_Map(function(x, y) 
                         tryCatch(fitzRoy::fetch_ladder(season = x, round_number = y, comp = "AFLM", source = "afltables"), 
                         error = function(e) NULL), 
                         x = years, y = rounds)
