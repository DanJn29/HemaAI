# HemaAI Backend

This backend provides:

- FastAPI REST endpoints under `/api/v1`
- deterministic rule-based CBC analysis
- ML inference endpoints backed by trained artifacts
- synthetic dataset generation and model training utilities

## Structure

```text
backend/
├── app/
├── scripts/
├── tests/
├── alembic/
├── alembic.ini
├── pyproject.toml
└── Dockerfile
```

## Local Setup

The backend reads environment variables from the repo-root `.env`.

```bash
cd backend
pip install -e .[dev]
alembic upgrade head
python -m app.seed.seed_data
uvicorn app.main:app --reload
```

## Tests

From `backend/`:

```bash
pytest
```

Or from the repo root:

```bash
docker-compose run --rm api pytest
```

## ML Scripts

Run from `backend/` and write artifacts to the repo-root `artifacts/` directory:

```bash
python scripts/generate_dataset.py --seed 42 --samples-per-class 250 --output-dir ../artifacts/datasets/run_name
python scripts/train_models.py --dataset-dir ../artifacts/datasets/run_name --dataset-variant default --output-dir ../artifacts/models/run_name --seed 42 --feature-modes raw_only hybrid
python scripts/select_best_model.py --output-dir ../artifacts/models/run_name
```

## Main Endpoints

- `POST /api/v1/analyses`
- `GET /api/v1/analyses/{id}`
- `POST /api/v1/recompute/{id}`
- `GET /api/v1/indicators`
- `GET /api/v1/diseases`
- `GET /api/v1/reference-ranges`
- `GET /api/v1/ml/model-info`
- `POST /api/v1/ml/predict`
- `POST /api/v1/ml/predict-and-compare`
