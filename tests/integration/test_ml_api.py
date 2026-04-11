from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.db import get_session_factory
from app.main import create_app
from app.ml.generation.dataset_builder import SyntheticDatasetBuilder
from app.ml.generation.generator import SyntheticGenerationConfig
from app.ml.inference.service import clear_ml_inference_caches
from app.ml.training.model_selection import write_best_model_metadata
from app.ml.training.pipeline import TrainingConfig, TrainingPipeline
from app.models.analysis_case import AnalysisCase
from app.models.analysis_result import AnalysisResult
from app.models.analysis_result_explanation import AnalysisResultExplanation
from app.models.analysis_value import AnalysisValue


FULL_CBC_PAYLOAD = {
    "sex": "female",
    "age": 28,
    "values": {
        "HGB": 109,
        "RBC": 3.9,
        "HCT": 0.33,
        "MCV": 72,
        "MCH": 23,
        "MCHC": 315,
        "RDW": 16.8,
        "PLT": 280,
        "WBC": 6.4,
        "NEU": 3.7,
        "LYM": 2.0,
        "MONO": 0.5,
        "EOS": 0.2,
        "BASO": 0.05,
    },
}


def test_ml_endpoints_predict_and_compare_without_persisting_analysis_records(
    db_session,
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    model_dir = tmp_path / "models"

    builder = SyntheticDatasetBuilder(
        session=db_session,
        config=SyntheticGenerationConfig(seed=42, samples_per_class=8),
    )
    bundle = builder.build()
    builder.export(dataset_dir, bundle)

    pipeline = TrainingPipeline(
        dataset_dir=dataset_dir,
        output_dir=model_dir,
        dataset_variant="default",
        config=TrainingConfig(
            seed=42,
            random_forest_estimators=20,
            catboost_iterations=20,
        ),
    )
    pipeline.run(feature_modes=["raw_only", "hybrid"])
    write_best_model_metadata(output_dir=model_dir, dataset_variants=("default",))

    monkeypatch.setenv("ML_BEST_MODEL_PATH", str(model_dir / "best_model.json"))
    get_settings.cache_clear()
    clear_ml_inference_caches()

    counts_before = _analysis_table_counts()
    app = create_app()
    with TestClient(app) as client:
        model_info_response = client.get("/api/v1/ml/model-info")
        assert model_info_response.status_code == 200
        model_info_body = model_info_response.json()
        assert model_info_body["model_name"]
        assert model_info_body["dataset_variant"] == "default"
        assert "validation_metrics" in model_info_body

        predict_response = client.post("/api/v1/ml/predict", json=FULL_CBC_PAYLOAD)
        assert predict_response.status_code == 200
        predict_body = predict_response.json()
        assert predict_body["predicted_label"]
        assert len(predict_body["top_3_predictions"]) >= 1
        assert predict_body["model_info"]["dataset_variant"] == "default"

        compare_response = client.post("/api/v1/ml/predict-and-compare", json=FULL_CBC_PAYLOAD)
        assert compare_response.status_code == 200
        compare_body = compare_response.json()
        assert compare_body["predicted_label"]
        assert compare_body["rule_engine"]["top1_label"]
        assert len(compare_body["rule_engine"]["top3_labels"]) >= 1

    counts_after = _analysis_table_counts()
    assert counts_after == counts_before


def test_ml_endpoints_return_not_found_when_best_model_artifact_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    missing_metadata_path = tmp_path / "missing-best-model.json"
    monkeypatch.setenv("ML_BEST_MODEL_PATH", str(missing_metadata_path))
    get_settings.cache_clear()
    clear_ml_inference_caches()

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/ml/model-info")
        assert response.status_code == 404
        assert "Best model metadata file was not found" in response.json()["detail"]


def _analysis_table_counts() -> dict[str, int]:
    session_factory = get_session_factory()
    with session_factory() as session:
        return {
            "analysis_cases": session.query(AnalysisCase).count(),
            "analysis_values": session.query(AnalysisValue).count(),
            "analysis_results": session.query(AnalysisResult).count(),
            "analysis_result_explanations": session.query(AnalysisResultExplanation).count(),
        }
