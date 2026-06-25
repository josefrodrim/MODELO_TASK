.PHONY: install test lint notebooks api docker clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

lint:
	python -m flake8 src/ --max-line-length=100 --ignore=E203,W503

notebooks:
	jupyter notebook notebooks/

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t modelo-riesgo:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

train-all:
	python -c "from src.models.train_pipeline import run_full_pipeline; run_full_pipeline()"
