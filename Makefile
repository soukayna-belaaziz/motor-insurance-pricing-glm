all:
	python -m src.pipeline

data:
	Rscript data/download_data.R

test:
	python -m pytest -q tests
	Rscript -e 'testthat::test_dir("tests/testthat", stop_on_failure = TRUE)'

.PHONY: all data test
