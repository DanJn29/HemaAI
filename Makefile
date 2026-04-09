PYTHON ?= python3

.PHONY: up down build migrate seed test logs run generate-dataset train-models

build:
	docker-compose build

up:
	docker-compose up -d db

run:
	docker-compose up --build

down:
	docker-compose down -v

migrate:
	docker-compose run --rm api alembic upgrade head

seed:
	docker-compose run --rm api python -m app.seed.seed_data

test:
	docker-compose up -d db
	docker-compose run --rm api pytest

logs:
	docker-compose logs -f api

generate-dataset:
	docker-compose run --rm api python scripts/generate_dataset.py --seed 42 --samples-per-class 100 --output-dir artifacts/datasets/latest

train-models:
	docker-compose run --rm api python scripts/train_models.py --dataset-dir artifacts/datasets/latest --dataset-variant default --output-dir artifacts/models/latest --seed 42 --feature-modes raw_only hybrid --include-rule-score-experiment
