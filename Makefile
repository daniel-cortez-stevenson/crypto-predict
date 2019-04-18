.PHONY: update_install install install_dev clean clean_pyc clean_test clean_build lint run_docker run_jupyter

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = crypto_predict
PYTHON_INTERPRETER = python3
PYTHON_VERSION = 3.5

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

update_install:
	pip install -U pip setuptools wheel

install: update_install
	pip install -U .

install_dev: update_install
	pip install -U -e . -r requirements.txt
	@echo ">>> Creating Jupyter Notebook kernel -> Python ($(PROJECT_NAME))"
	$(PYTHON_INTERPRETER) -m ipykernel install --sys-prefix --name $(PROJECT_NAME) --display-name "Python ($(PROJECT_NAME))"
	jupyter contrib nbextension install --sys-prefix

run_jupyter:
	jupyter notebook --config ./notebooks/jupyter_notebook_config.py

test_all: clean_test test test_notebooks

test: clean_test
	bash -c "coverage run setup.py test"

test_notebooks: clean_test
	$(PYTHON_INTERPRETER) -m pytest --nbval-lax -n auto --max-worker-restart 1 --dist loadscope ./notebooks

run_docker: clean
	docker build -f ./docker/Dockerfile -t crypr-api .
	docker run -p 5000:5000 crypr-api

clean: clean_pyc clean_test clean_build clean_logs

clean_build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean_logs:
	find . -name 'logs' -exec rm -fr {} +

clean_pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean_test: clean_build
	rm -fr .tox/
	rm -f .coverage
	rm -fr .pytest_cache

lint:
	$(PYTHON_INTERPRETER) -m flake8 --exit-zero ./crypr
	$(PYTHON_INTERPRETER) -m flake8 --exit-zero ./notebooks

dist: clean
	$(PYTHON_INTERPRETER) setup.py sdist
	$(PYTHON_INTERPRETER) setup.py bdist_wheel
	ls -l dist

create_environment:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
	conda create -y --name $(PROJECT_NAME) python=$(PYTHON_VERSION)
	@echo ">>> New conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
else
	pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	|   more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
