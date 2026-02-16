test_run:
	Rscript -e 'testthat::test_dir("tests/R")'

scrape_data:
	Rscript scripts/data_pull.r

run_model:
	python scripts/pipeline.py

run_model_refactored:
	python scripts/pipeline.py