setwd("/afl_probabilistic_model")

testthat::test_that("Ladder data pulled", {
    source("R/data_scrape.r")
    test_loc = "temp"
    dir.create(test_loc, recursive = TRUE)
    write_raw_data(table_type = "ladder", season = 2024, round = c(1,2), output_path = test_loc)
    files_created = length(list.files(paste0(test_loc,"/Season=2024")))
    
    df <- arrow::read_parquet(paste0(test_loc, "/Season=2024/part-0.parquet"))
    number_rounds <- length(unique(df[["Round.Number"]]))

    unlink(test_loc, recursive = TRUE)

    testthat::expect_equal(files_created, 1)
    testthat::expect_equal(number_rounds, 2)
})

testthat::test_that("Results data pulled", {
    source("R/data_scrape.r")
    test_loc = "temp"
    dir.create(test_loc, recursive = TRUE)
    write_raw_data(table_type = "results", season = 2024, round = c(1,2), output_path = test_loc)
    files_created = length(list.files(paste0(test_loc,"/Season=2024")))
    
    df <- arrow::read_parquet(paste0(test_loc, "/Season=2024/part-0.parquet"))
    number_rounds <- length(unique(df[["Round.Number"]]))

    unlink(test_loc, recursive = TRUE)

    testthat::expect_equal(files_created, 1)
    testthat::expect_equal(number_rounds, 2)
})
