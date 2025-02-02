scrape_ladder <- function(season, round, path) {
    df <- tryCatch(fitzRoy::fetch_ladder(season = season, 
                                         round_number = round, 
                                         comp = "AFLM", 
                                         source = "afltables"
                                         ), 
            error = function(e) NULL)

    if(!is.null(df)){
        arrow::write_dataset(dataset = df, 
                             format = "parquet",
                             path = path, 
                             partitioning = "Season",
                             existing_data_behavior = "error")
    }
}