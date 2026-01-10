.PHONY: env

env:
	conda env create -f environment.yml || conda env update -f environment.yml
	conda run -n DROPOUT pip install -e .