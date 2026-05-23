import json
from pathlib import Path

import joblib
import pytest

from app.core.config import get_settings
from app.core.exceptions import NotFoundError
from app.ml.generation.dataset_builder import SyntheticDatasetBuilder
from app.ml.generation.generator import SyntheticGenerationConfig
from app.ml.inference.service import (
    MLInferenceService,
    _load_best_model_metadata,
    _load_model_bundle,
    clear_ml_inference_caches,
)
from app.ml.training.model_selection import (
    ModelSelectionError,
    select_best_model,
    write_best_model_metadata,
)
from app.ml.training.pipeline import TrainingConfig, TrainingPipeline
from app.schemas.analysis import AnalysisCreateRequest


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


def test_model_selection_prefers_default_candidates_for_deployment(tmp_path: Path) -> None:
    _write_dummy_model_artifact(tmp_path / "strict" / "hybrid" / "catboost.joblib")
    _write_dummy_model_artifact(tmp_path / "default" / "hybrid" / "catboost.joblib")
    _write_comparison_artifact(
        tmp_path,
        "strict",
        {
            "hybrid": {
                "feature_mode": "hybrid",
                "include_rule_scores": False,
                "models": {
                    "catboost": _build_metrics(
                        validation_accuracy=1.0,
                        validation_f1=1.0,
                        test_accuracy=1.0,
                        test_f1=1.0,
                    )
                },
            }
        },
    )
    _write_comparison_artifact(
        tmp_path,
        "default",
        {
            "hybrid": {
                "feature_mode": "hybrid",
                "include_rule_scores": False,
                "models": {
                    "catboost": _build_metrics(
                        validation_accuracy=0.98,
                        validation_f1=0.97,
                        test_accuracy=0.96,
                        test_f1=0.95,
                    )
                },
            }
        },
    )

    selected = select_best_model(output_dir=tmp_path)

    assert selected.dataset_variant == "default"
    assert selected.feature_mode == "hybrid"
    assert selected.model_name == "catboost"


def test_model_selection_prefers_hybrid_on_true_metric_tie(tmp_path: Path) -> None:
    _write_dummy_model_artifact(tmp_path / "default" / "raw_only" / "random_forest.joblib")
    _write_dummy_model_artifact(tmp_path / "default" / "hybrid" / "catboost.joblib")
    identical_metrics = _build_metrics(
        validation_accuracy=0.91,
        validation_f1=0.9,
        test_accuracy=0.9,
        test_f1=0.89,
    )
    _write_comparison_artifact(
        tmp_path,
        "default",
        {
            "raw_only": {
                "feature_mode": "raw_only",
                "include_rule_scores": False,
                "models": {"random_forest": identical_metrics},
            },
            "hybrid": {
                "feature_mode": "hybrid",
                "include_rule_scores": False,
                "models": {"catboost": identical_metrics},
            },
        },
    )

    selected = select_best_model(output_dir=tmp_path, dataset_variants=("default",))

    assert selected.feature_mode == "hybrid"
    assert selected.model_name == "catboost"


def test_model_selection_falls_back_to_strict_when_default_has_no_valid_candidates(
    tmp_path: Path,
) -> None:
    _write_dummy_model_artifact(tmp_path / "strict" / "raw_only" / "random_forest.joblib")
    _write_comparison_artifact(
        tmp_path,
        "default",
        {
            "hybrid": {
                "feature_mode": "hybrid",
                "include_rule_scores": True,
                "models": {
                    "catboost": _build_metrics(
                        validation_accuracy=0.98,
                        validation_f1=0.98,
                        test_accuracy=0.97,
                        test_f1=0.97,
                    )
                },
            }
        },
    )
    _write_comparison_artifact(
        tmp_path,
        "strict",
        {
            "raw_only": {
                "feature_mode": "raw_only",
                "include_rule_scores": False,
                "models": {
                    "random_forest": _build_metrics(
                        validation_accuracy=0.95,
                        validation_f1=0.95,
                        test_accuracy=0.94,
                        test_f1=0.94,
                    )
                },
            }
        },
    )

    selected = select_best_model(output_dir=tmp_path)

    assert selected.dataset_variant == "strict"
    assert selected.feature_mode == "raw_only"
    assert selected.model_name == "random_forest"


def test_model_selection_rejects_missing_artifacts_and_malformed_comparisons(
    tmp_path: Path,
) -> None:
    _write_dummy_model_artifact(tmp_path / "strict" / "hybrid" / "catboost.joblib")
    (tmp_path / "default_comparison.json").write_text("{not-json", encoding="utf-8")
    _write_comparison_artifact(
        tmp_path,
        "strict",
        {
            "raw_only": {
                "feature_mode": "raw_only",
                "include_rule_scores": False,
                "models": {
                    "random_forest": _build_metrics(
                        validation_accuracy=0.98,
                        validation_f1=0.97,
                        test_accuracy=0.97,
                        test_f1=0.96,
                    )
                },
            },
            "hybrid": {
                "feature_mode": "hybrid",
                "include_rule_scores": False,
                "models": {
                    "catboost": _build_metrics(
                        validation_accuracy=0.99,
                        validation_f1=0.98,
                        test_accuracy=0.98,
                        test_f1=0.97,
                    )
                },
            },
        },
    )

    selected = select_best_model(output_dir=tmp_path)

    assert selected.dataset_variant == "strict"
    assert selected.feature_mode == "hybrid"
    assert selected.model_name == "catboost"


def test_model_selection_rejects_candidates_missing_required_metrics(tmp_path: Path) -> None:
    _write_dummy_model_artifact(tmp_path / "default" / "hybrid" / "catboost.joblib")
    _write_comparison_artifact(
        tmp_path,
        "default",
        {
            "hybrid": {
                "feature_mode": "hybrid",
                "include_rule_scores": False,
                "models": {
                    "catboost": {
                        "accuracy": 0.9,
                        "precision_macro": 0.9,
                        "recall_macro": 0.9,
                        "top3_accuracy": 0.95,
                        "validation_metrics": {
                            "accuracy": 0.91,
                            "precision_macro": 0.9,
                            "recall_macro": 0.9,
                        },
                    }
                },
            }
        },
    )

    with pytest.raises(ModelSelectionError, match="missing required"):
        select_best_model(output_dir=tmp_path, dataset_variants=("default",))


def test_write_best_model_metadata_does_not_overwrite_on_selection_failure(
    tmp_path: Path,
) -> None:
    existing_metadata = tmp_path / "best_model.json"
    existing_metadata.write_text('{"keep":"me"}', encoding="utf-8")
    (tmp_path / "default_comparison.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(ModelSelectionError, match="No valid deployable model candidates"):
        write_best_model_metadata(output_dir=tmp_path, dataset_variants=("default",))

    assert existing_metadata.read_text(encoding="utf-8") == '{"keep":"me"}'


def test_metadata_loader_raises_not_found_for_missing_or_broken_artifacts(tmp_path: Path) -> None:
    clear_ml_inference_caches()
    with pytest.raises(NotFoundError):
        _load_best_model_metadata(str(tmp_path / "missing.json"))

    metadata_path = tmp_path / "best_model.json"
    metadata_path.write_text(
        json.dumps(
            {
                "selection_rule": "test",
                "artifact_format_version": 1,
                "model_name": "catboost",
                "dataset_variant": "default",
                "feature_mode": "hybrid",
                "include_rule_scores": False,
                "model_path": "default/hybrid/catboost.joblib",
                "comparison_path": "default_comparison.json",
                "validation_metrics": {},
                "test_metrics": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(NotFoundError):
        _load_best_model_metadata(str(metadata_path))


def test_metadata_and_model_bundle_are_cached_process_wide(tmp_path: Path) -> None:
    artifact_path = tmp_path / "default" / "raw_only" / "random_forest.joblib"
    _write_dummy_model_artifact(artifact_path)
    metadata_path = tmp_path / "best_model.json"
    metadata_path.write_text(
        json.dumps(
            {
                "selection_rule": "test",
                "artifact_format_version": 1,
                "model_name": "random_forest",
                "dataset_variant": "default",
                "feature_mode": "raw_only",
                "include_rule_scores": False,
                "model_path": "default/raw_only/random_forest.joblib",
                "comparison_path": "default_comparison.json",
                "validation_metrics": {},
                "test_metrics": {},
            }
        ),
        encoding="utf-8",
    )

    clear_ml_inference_caches()
    first_metadata = _load_best_model_metadata(str(metadata_path))
    second_metadata = _load_best_model_metadata(str(metadata_path))
    first_bundle = _load_model_bundle(first_metadata["_resolved_model_path"])
    second_bundle = _load_model_bundle(first_metadata["_resolved_model_path"])

    assert first_metadata is second_metadata
    assert first_bundle is second_bundle


def test_hybrid_inference_reuses_runtime_feature_building(db_session, tmp_path: Path, monkeypatch) -> None:
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
    results = pipeline.run(feature_modes=["hybrid"])
    hybrid_metrics = results["experiments"]["hybrid"]["models"]["catboost"]

    metadata_path = model_dir / "best_model.json"
    metadata_path.write_text(
        json.dumps(
            {
                "selection_rule": "unit-test",
                "artifact_format_version": 1,
                "model_name": "catboost",
                "dataset_variant": "default",
                "feature_mode": "hybrid",
                "include_rule_scores": False,
                "model_path": "default/hybrid/catboost.joblib",
                "comparison_path": "default_comparison.json",
                "validation_metrics": hybrid_metrics["validation_metrics"],
                "test_metrics": {
                    key: value
                    for key, value in hybrid_metrics.items()
                    if key != "validation_metrics"
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("ML_BEST_MODEL_PATH", str(metadata_path))
    get_settings.cache_clear()
    clear_ml_inference_caches()

    service = MLInferenceService(db_session, settings=get_settings())
    payload = AnalysisCreateRequest.model_validate(FULL_CBC_PAYLOAD)
    prediction = service.predict(payload)

    assert prediction["predicted_label"]
    assert prediction["model_info"]["feature_mode"] == "hybrid"
    assert len(prediction["top_3_predictions"]) >= 1
    assert all("probability" in item for item in prediction["top_3_predictions"])


def _write_dummy_model_artifact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": "model",
            "preprocessor": "preprocessor",
            "label_encoder": "label_encoder",
            "feature_mode": "raw_only",
            "include_rule_scores": False,
        },
        path,
    )


def _write_comparison_artifact(
    base_dir: Path,
    dataset_variant: str,
    experiments: dict[str, object],
) -> None:
    (base_dir / f"{dataset_variant}_comparison.json").write_text(
        json.dumps(
            {
                "dataset_variant": dataset_variant,
                "experiments": experiments,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_metrics(
    *,
    validation_accuracy: float,
    validation_f1: float,
    test_accuracy: float,
    test_f1: float,
) -> dict[str, object]:
    return {
        "accuracy": test_accuracy,
        "f1_macro": test_f1,
        "precision_macro": test_f1,
        "recall_macro": test_f1,
        "top3_accuracy": 1.0,
        "validation_metrics": {
            "accuracy": validation_accuracy,
            "f1_macro": validation_f1,
            "precision_macro": validation_f1,
            "recall_macro": validation_f1,
            "top3_accuracy": 1.0,
        },
    }
