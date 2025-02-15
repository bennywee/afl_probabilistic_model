source('R/data_type_lookup.r')

scrape_data <- function(scraping_function, season, round) {
    tryCatch(scraping_function(season = season, 
                           round_number = round, 
                           comp = "AFLM", 
                           source = "afltables"), 
            error = function(e){print(e); return(NULL)})

}

write_raw_data <- function(table_type, season, rounds, output_path) {
    lookup_function <- data_type_lookup[[table_type]]

    data_ls <- future.apply::future_Map(function(x, y) 
                         scrape_data(scraping_function = lookup_function[[1]], season = x, round = y),
                         x = season, y = rounds)

    df <- do.call(rbind, data_ls) |>
        arrow::arrow_table(schema = lookup_function[[2]])

    arrow::write_dataset(dataset = df, 
                         format = "parquet",
                         path = output_path, 
                         partitioning = "Season",
                         existing_data_behavior = "overwrite")
}
