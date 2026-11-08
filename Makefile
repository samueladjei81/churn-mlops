# Every command sets PYTHONPATH itself, so `No module named 'churn'` can't happen.
PY ?= python
export PYTHONPATH := src

.PHONY: help install install-mlflow test simulate setup train month status score api mlflow \
        check-features docker-build docker-run clean

help:
	@echo "make install         install the project (once, inside your .venv)"
	@echo "make install-mlflow  optional: experiment tracking dashboard"
	@echo "make test            run the tests"
	@echo "make simulate        run the WHOLE pipeline end to end (about 2 minutes)"
	@echo "make status          show model versions and monthly monitoring"
	@echo "make api             start the API on http://localhost:8000/docs"
	@echo "make mlflow          open the MLflow dashboard on http://localhost:5000"
	@echo "make score IN=file.csv OUT=scores.csv   score your own CSV"
	@echo "make setup / make train / make month M=1   run individual steps"
	@echo "make clean           delete generated data, models and reports"

install:
	pip install -e ".[api,dev]"

install-mlflow:
	pip install mlflow

test:
	$(PY) -m pytest -q

simulate:
	$(PY) -m churn.cli simulate
	@echo "Open reports/summary.html in your browser."

setup:
	$(PY) -m churn.cli setup

train:
	$(PY) -m churn.cli train

month:
	$(PY) -m churn.cli run-month $(M)

status:
	$(PY) -m churn.cli status

score:
	$(PY) -m churn.cli score --input "$(IN)" --output "$(OUT)"

api:
	$(PY) -m uvicorn churn.api:app --reload --port 8000

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

check-features:
	$(PY) scripts/check_feature_dependence.py

docker-build:
	docker build -t churn-api .

docker-run:
	docker run --rm -p 8000:8000 churn-api

clean:
	rm -rf data/churn.db data/batches models reports artifacts/*.csv artifacts/*.json
