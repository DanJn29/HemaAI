# HemaAI

HemaAI is organized as a split repo with a Python backend, a React frontend, and root-level ML artifacts:

- `backend/` contains the FastAPI API, rule engine, ML pipeline, migrations, scripts, and tests.
- `frontend/` contains the React + TypeScript dashboard for CBC entry and ML/rule comparison.
- `artifacts/` remains at the project root for generated datasets, trained models, metrics, and deployable model metadata.

## Repo Layout

```text
.
├── backend/
├── frontend/
├── artifacts/
├── docker-compose.yml
├── Makefile
└── .env
```

## Run Everything With Docker

Create a local env file first:

```bash
cp .env.example .env
```

Then start the stack:

```bash
docker-compose up --build
```

Services:

- Backend API: `http://localhost:8000`
- Frontend UI: `http://localhost:5174`

The frontend talks to the backend through the Vite dev proxy. The backend still reads and writes model/dataset artifacts from the root `artifacts/` directory.

## Backend Only

```bash
cd backend
pip install -e .[dev]
uvicorn app.main:app --reload
```

See [backend/README.md](backend/README.md) for backend operations, migrations, tests, and ML pipeline commands.

## Frontend Only

```bash
cd frontend
npm install
FRONTEND_PORT=5174 npm run dev -- --port 5174
```

The frontend defaults to `http://localhost:8000` as the local API proxy target.

## Common Commands

```bash
make build
make run
make test
make migrate
make seed
make frontend-build
```
