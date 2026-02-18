test_run:
	Rscript -e 'testthat::test_dir("tests/R")'

test_python:
	python -m pytest tests/test_modelling_*.py -v

test_python_cov:
	python -m pytest tests/test_modelling_*.py -v --cov=src.modelling --cov-report=html

scrape_data:
	Rscript scripts/data_pull.r

run_model:
	python -m src.modelling.run_train_pred
