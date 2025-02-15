test_run:
	Rscript -e 'testthat::test_dir("tests/R")'

scrape_data:
	Rscript scripts/data_pull.r