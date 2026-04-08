PYTHON ?= python3

.PHONY: up down build migrate seed test logs run

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

