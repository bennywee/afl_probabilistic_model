testthat::test_that("Ladder data pulled", {
    source("../../R/data_scrape.r")
    test_loc = "tests/R/temp"
    dir.create(test_loc, recursive = TRUE)
    scrape_ladder(season = 2024, round = 1, path = test_loc)
    files_created = length(list.files(paste0(test_loc,"/Season=2024")))
    unlink(test_loc, recursive = TRUE)

    testthat::expect_equal(files_created, 1)
})
