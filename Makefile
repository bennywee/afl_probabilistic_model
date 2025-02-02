test_run:
	Rscript -e 'testthat::test_dir("tests/R")'

scrape_ladder:
	Rscript scripts/data_pull.r