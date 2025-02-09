scrape_ladder <- function(season, round) {
    tryCatch(fitzRoy::fetch_ladder(season = season, 
                                         round_number = round, 
                                         comp = "AFLM", 
                                         source = "afltables"
                                         ), 
            error = function(e) NULL)

}

scrape_write_ladder <- function(season, rounds, output_path) {
    data_ls <- future.apply::future_Map(function(x, y) 
                         tryCatch(scrape_ladder(season = x, round = y), 
                         error = function(e){print(e); return(NULL)}),
                         x = season, y = rounds)

    df <- do.call(rbind, data_ls)

    arrow::write_dataset(dataset = df, 
                         format = "parquet",
                         path = output_path, 
                         partitioning = "Season",
                         existing_data_behavior = "overwrite")
}
