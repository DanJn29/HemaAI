# HemaAI MVP

HemaAI is a backend-first MVP for AI-assisted blood test interpretation. It accepts manually entered CBC-style lab values, compares them against adult sex- and age-aware reference ranges, applies a rule-based scoring engine plus multi-indicator pattern bonuses, and returns ranked disease hypotheses with explanations.

This project is a clinical decision-support prototype. It does not provide a definitive diagnosis, does not replace a clinician, and uses demo seed data for educational development purposes.

## Secrets Handling

- Real credentials should live in a local `.env` file, not in committed source.
- Copy `.env.example` to `.env` and adjust the values for your machine or environment.
- `.env` and other local env files are ignored by both Git and Docker build context so they do not get pushed or baked into images accidentally.
- The tracked files now require those env vars instead of falling back to committed DB credentials.

## Architecture

- `FastAPI` exposes REST endpoints under `/api/v1`.
- `SQLAlchemy 2.x` models the PostgreSQL schema.
- `Alembic` manages migrations.
- `app/services` contains the interpretation and scoring engine:
  - `ReferenceRangeService`
  - `DeviationInterpreter`
  - `RuleScoringService`
  - `PatternMatchingService`
  - `AnalysisOrchestrator`
- `app/repositories` keeps route handlers thin and centralizes common persistence queries.
- `app/seed/seed_data.py` populates indicators, diseases, reference ranges, rules, and patterns.
- `app/ml` contains the synthetic dataset generation and ML training pipeline built on top of the existing rule-based engine.
- `pytest` covers unit logic and integration flows against PostgreSQL.

## Database Overview

Core catalog tables:

- `indicators`
- `reference_ranges`
- `deviation_states`
- `diseases`
- `indicator_rules`
- `pattern_rules`
- `pattern_rule_conditions`

Analysis tables:

- `analysis_cases`
- `analysis_values`
- `analysis_results`
- `analysis_result_explanations`

Minimal schema extension:

- `pattern_rule_conditions.match_mode`: `exact | family`
- `pattern_rule_conditions.deviation_family`: `low | high | normal`

This extension allows patterns such as “any low HGB” without hardcoding multiple exact deviation-state rows.

## Deviation Semantics

Classification order is explicit and gap-free:

1. `normal_min <= value <= normal_max` => `normal`
2. `value < normal_min` => low-side severity
3. `value > normal_max` => high-side severity

Low-side severity:

- `value <= severe_low_threshold` => `severe_low`
- `value <= moderate_low_threshold` => `moderate_low`
- otherwise => `mild_low`

High-side severity:

- `value >= severe_high_threshold` => `severe_high`
- `value >= moderate_high_threshold` => `moderate_high`
- otherwise => `mild_high`

`mild_low_threshold` and `mild_high_threshold` are stored to preserve the requested schema, but the engine treats `normal_min` and `normal_max` as the definitive inclusive normal boundaries.

## Seed Data Scope

- Adult-only demo ranges for ages `18-120`
- Sex- and age-stratified ranges for `male` and `female`
- Exact adult buckets for every indicator:
  - `18-40`
  - `41-65`
  - `66-120`
- Minimum indicator set:
  - `WBC`, `RBC`, `HGB`, `HCT`, `MCV`, `MCH`, `MCHC`, `PLT`, `RDW`, `NEU`, `LYM`, `MONO`, `EOS`, `BASO`
- Disease hypotheses:
  - `normal`
  - `iron_deficiency_anemia`
  - `macrocytic_anemia`
  - `bacterial_infection`
  - `viral_infection`
  - `allergic_or_parasitic_pattern`
  - `thrombocytopenia_pattern`
  - `hematologic_malignancy_suspicion`

## Engine Behavior

- Indicator-level rules support or contradict diseases with weighted evidence.
- Pattern rules apply bonuses when all conditions match.
- `normalized_score` is stored on `analysis_values` as informational ML-readiness metadata only. It does not affect MVP disease scoring.
- Normal fallback is threshold-based:
  - if no non-normal disease reaches `MIN_PATHOLOGY_SCORE`
  - and no matched pattern reaches `STRONG_PATTERN_BONUS_THRESHOLD`
  - the response returns `normal` as the fallback hypothesis

Default engine settings are defined in `app/core/config.py`:

- `MIN_PATHOLOGY_SCORE = 3.0`
- `STRONG_PATTERN_BONUS_THRESHOLD = 3.0`
- `MIN_PERSISTED_SCORE = 1.0`
- `MAX_PERSISTED_NON_NORMAL = 5`
- `MAX_RETURNED_HYPOTHESES = 3`

## Local Setup

### Option 1: Docker Compose

Create a local env file first:

```bash
cp .env.example .env
```

Edit `.env` and set your own local database values.

Start the stack:

```bash
make run
```

The API container runs migrations, seeds the database, and serves FastAPI on `http://localhost:8000`.

Stop everything:

```bash
make down
```

### Option 2: Local Python environment

Create a local env file:

```bash
cp .env.example .env
```

Install dependencies:

```bash
pip install -e .[dev]
```

Run PostgreSQL separately, then set component env vars from your local `.env`:

```bash
export DB_HOST=...
export DB_PORT=...
export DB_NAME=...
export DB_USER=...
export DB_PASSWORD=...
export TEST_DB_HOST=...
export TEST_DB_PORT=...
export TEST_DB_NAME=...
export TEST_DB_USER=...
export TEST_DB_PASSWORD=...
```

Apply migrations:

```bash
alembic upgrade head
```

Seed demo data:

```bash
python -m app.seed.seed_data
```

Run the API:

```bash
uvicorn app.main:app --reload
```

## Migrations

Apply migrations in Docker:

```bash
make migrate
```

Generate a future migration manually if you extend the schema:

```bash
alembic revision -m "describe change"
```

## Seeding

Populate demo catalog data:

```bash
make seed
```

The seed script replaces all `reference_ranges` rows in one transaction and repopulates them with the current age-stratified adult demo dataset.

## Environment Variables

The app can be configured either with a full URL override or with component-based settings.

Preferred component-based settings:

- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`
- `TEST_DB_HOST`
- `TEST_DB_PORT`
- `TEST_DB_NAME`
- `TEST_DB_USER`
- `TEST_DB_PASSWORD`

Optional full URL overrides:

- `DATABASE_URL`
- `TEST_DATABASE_URL`

## Running Tests

Run all tests against PostgreSQL in Docker:

```bash
make test
```

The test suite resets the PostgreSQL test schema, runs Alembic migrations, seeds demo data, and then executes both unit and integration tests.

## Synthetic Dataset and ML Pipeline

The project also includes a synthetic dataset generation and ML training pipeline for educational prototyping. It does not use real clinical cases.

- Synthetic cases are generated from the existing sex- and age-aware reference ranges.
- The existing backend services are reused for deviation interpretation, rule scoring, and pattern matching.
- The primary supervised target is the synthetic case `intended_label`.
- Rule-engine outputs are used for validation and quality labeling only.

Generate a balanced synthetic dataset:

```bash
python scripts/generate_dataset.py --seed 42 --samples-per-class 1000 --output-dir artifacts/datasets/run_name
```

Or via Docker:

```bash
make generate-dataset
```

Train and evaluate models:

```bash
python scripts/train_models.py --dataset-dir artifacts/datasets/run_name --dataset-variant default --output-dir artifacts/models/run_name --seed 42 --feature-modes raw_only hybrid --include-rule-score-experiment
```

Or via Docker:

```bash
make train-models
```

The dataset export includes:

- `all_cases.csv`
- `good_cases.csv`
- `ambiguous_cases.csv`
- `bad_cases.csv`
- `train_dataset_strict.csv`
- `train_dataset_default.csv`
- split files for each training dataset variant
- `dataset_summary.json`
- `dataset_diagnostics.json`

The training pipeline exports model artifacts, metrics, feature importance reports, and rule-engine-vs-ML comparison summaries under `artifacts/models/`.
When using Docker, the `artifacts/` directory is bind-mounted so the generated datasets and model outputs are written back to the host project directory.

## API Endpoints

- `POST /api/v1/analyses`
- `GET /api/v1/analyses/{id}`
- `GET /api/v1/indicators`
- `GET /api/v1/diseases`
- `GET /api/v1/reference-ranges`
- `POST /api/v1/recompute/{id}`

## Sample Request

```bash
curl -X POST http://localhost:8000/api/v1/analyses \
  -H "Content-Type: application/json" \
  -d '{
    "sex": "female",
    "age": 28,
    "values": {
      "HGB": 109,
      "MCV": 72,
      "MCH": 23,
      "RDW": 16.8,
      "RBC": 3.9
    }
  }'
```

Example response shape:

```json
{
  "analysis_id": 123,
  "patient": {
    "sex": "female",
    "age": 28
  },
  "indicator_interpretation": [
    {
      "indicator_code": "HGB",
      "raw_value": 109,
      "deviation_state": "moderate_low"
    }
  ],
  "top_hypotheses": [
    {
      "rank": 1,
      "disease_code": "iron_deficiency_anemia",
      "disease_name": "Iron Deficiency Anemia",
      "total_score": 11.5,
      "confidence": null,
      "explanations": [
        {
          "type": "indicator_rule",
          "text": "Low HGB supported iron deficiency anemia.",
          "score_effect": 3.0
        },
        {
          "type": "pattern_rule",
          "text": "Pattern match: low HGB, low MCV, low MCH, and high RDW supported iron deficiency anemia.",
          "score_effect": 4.0
        }
      ]
    }
  ],
  "disclaimer": "This system provides ranked hypotheses for educational decision-support purposes and is not a definitive medical diagnosis."
}
```

## Limitations

- The MVP is adult-only and rejects ages outside `18-120`.
- Reference ranges and rules are demo seed data, not validated clinical guidelines.
- The engine is CBC-oriented and does not infer a definitive diagnosis.
- Confidence scoring is not implemented; `confidence` remains `null`.
- No ML model is active yet, though `normalized_score` and explanation persistence are ready for future ML augmentation.

## Disclaimer

This software returns ranked hypotheses for educational and decision-support purposes only. It is not a diagnostic device and must not be used as a sole basis for clinical decisions.
