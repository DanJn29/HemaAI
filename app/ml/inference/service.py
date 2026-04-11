from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import DomainValidationError, NotFoundError
from app.ml.generation.evaluator import RuleEngineContext, RuleEngineEvaluator
from app.ml.training.features import INDICATOR_COLUMNS, build_feature_set
from app.schemas.analysis import AnalysisCreateRequest


def clear_ml_inference_caches() -> None:
    _load_best_model_metadata.cache_clear()
    _load_model_bundle.cache_clear()


class MLInferenceService:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self._runtime_context: RuleEngineContext | None = None

    def get_model_info(self) -> dict[str, Any]:
        metadata = self._get_best_model_metadata()
        return _public_model_metadata(metadata)

    def predict(self, payload: AnalysisCreateRequest) -> dict[str, Any]:
        prediction, _ = self._predict_with_runtime(payload)
        return prediction

    def predict_and_compare(self, payload: AnalysisCreateRequest) -> dict[str, Any]:
        prediction, runtime_evaluation = self._predict_with_runtime(payload)
        return {
            **prediction,
            "rule_engine": {
                "top1_label": runtime_evaluation.top1_label,
                "top3_labels": runtime_evaluation.top3_labels,
            },
        }

    def _predict_with_runtime(
        self,
        payload: AnalysisCreateRequest,
    ) -> tuple[dict[str, Any], Any]:
        metadata = self._get_best_model_metadata()
        bundle = _load_model_bundle(metadata["_resolved_model_path"])
        raw_values = self._validated_raw_values(payload)
        feature_frame, runtime_evaluation = self._build_feature_frame(
            payload=payload,
            raw_values=raw_values,
            feature_mode=str(metadata["feature_mode"]),
            include_rule_scores=bool(metadata.get("include_rule_scores", False)),
        )

        transformed = bundle["preprocessor"].transform(feature_frame)
        probabilities = bundle["model"].predict_proba(transformed)[0]
        label_encoder = bundle["label_encoder"]
        top_indices = np.argsort(probabilities)[::-1][: min(3, len(probabilities))]
        top_predictions = [
            {
                "label": str(label_encoder.inverse_transform([int(index)])[0]),
                "probability": float(probabilities[int(index)]),
            }
            for index in top_indices
        ]

        return (
            {
                "predicted_label": top_predictions[0]["label"],
                "top_3_predictions": top_predictions,
                "model_info": _public_prediction_model_info(metadata),
            },
            runtime_evaluation,
        )

    def _build_feature_frame(
        self,
        *,
        payload: AnalysisCreateRequest,
        raw_values: dict[str, Decimal],
        feature_mode: str,
        include_rule_scores: bool,
    ) -> tuple[pd.DataFrame, Any]:
        evaluator = RuleEngineEvaluator(
            self.session,
            context=self._get_runtime_context(),
        )
        row, runtime_evaluation = evaluator.build_runtime_feature_row(
            sex=payload.sex,
            age=payload.age,
            raw_values=raw_values,
        )
        feature_set = build_feature_set(
            pd.DataFrame([row]),
            feature_mode=feature_mode,
            include_rule_scores=include_rule_scores,
        )
        return feature_set.frame, runtime_evaluation

    def _get_runtime_context(self) -> RuleEngineContext:
        if self._runtime_context is None:
            self._runtime_context = RuleEngineContext.from_session(
                self.session,
                preload_reference_cache=False,
            )
        return self._runtime_context

    def _get_best_model_metadata(self) -> dict[str, Any]:
        return _load_best_model_metadata(self.settings.ml_best_model_path)

    @staticmethod
    def _validated_raw_values(payload: AnalysisCreateRequest) -> dict[str, Decimal]:
        submitted_values = {
            item.indicator_code: item.raw_value
            for item in payload.values
        }
        unknown_codes = sorted(set(submitted_values) - set(INDICATOR_COLUMNS))
        if unknown_codes:
            raise DomainValidationError(
                f"Unknown indicator codes: {', '.join(unknown_codes)}"
            )

        missing_codes = [code for code in INDICATOR_COLUMNS if code not in submitted_values]
        if missing_codes:
            raise DomainValidationError(
                "ML prediction requires all supported CBC indicators. "
                f"Missing: {', '.join(missing_codes)}"
            )

        return {
            code: submitted_values[code]
            for code in INDICATOR_COLUMNS
        }


@lru_cache(maxsize=4)
def _load_best_model_metadata(metadata_path: str) -> dict[str, Any]:
    resolved_metadata_path = _resolve_path(Path(metadata_path))
    if not resolved_metadata_path.exists():
        raise NotFoundError(
            f"Best model metadata file was not found at {resolved_metadata_path}."
        )

    try:
        metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NotFoundError(
            f"Best model metadata at {resolved_metadata_path} could not be read."
        ) from exc

    required_keys = {
        "model_name",
        "dataset_variant",
        "feature_mode",
        "model_path",
    }
    missing_keys = sorted(required_keys - set(metadata))
    if missing_keys:
        raise NotFoundError(
            "Best model metadata at "
            f"{resolved_metadata_path} is unreadable or missing required fields: {', '.join(missing_keys)}."
        )
    model_path = metadata["model_path"]

    resolved_model_path = _resolve_relative_path(
        base_path=resolved_metadata_path.parent,
        candidate_path=Path(str(model_path)),
    )
    if not resolved_model_path.exists():
        raise NotFoundError(
            f"Trained model artifact referenced by {resolved_metadata_path} was not found at {resolved_model_path}."
        )

    return {
        **metadata,
        "_resolved_metadata_path": str(resolved_metadata_path),
        "_resolved_model_path": str(resolved_model_path),
    }


@lru_cache(maxsize=4)
def _load_model_bundle(model_path: str) -> dict[str, Any]:
    resolved_model_path = _resolve_path(Path(model_path))
    if not resolved_model_path.exists():
        raise NotFoundError(
            f"Trained model artifact was not found at {resolved_model_path}."
        )

    try:
        bundle = joblib.load(resolved_model_path)
    except Exception as exc:
        raise NotFoundError(
            f"Trained model artifact at {resolved_model_path} could not be read."
        ) from exc
    required_keys = {"model", "preprocessor", "label_encoder", "feature_mode", "include_rule_scores"}
    missing_keys = sorted(required_keys - set(bundle))
    if missing_keys:
        raise NotFoundError(
            "Trained model artifact at "
            f"{resolved_model_path} is unreadable or missing required fields: {', '.join(missing_keys)}."
        )
    return bundle


def _public_prediction_model_info(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": metadata["model_name"],
        "dataset_variant": metadata["dataset_variant"],
        "feature_mode": metadata["feature_mode"],
        "include_rule_scores": bool(metadata.get("include_rule_scores", False)),
    }


def _public_model_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        **_public_prediction_model_info(metadata),
        "selection_rule": metadata["selection_rule"],
        "artifact_format_version": metadata["artifact_format_version"],
        "model_path": metadata["model_path"],
        "comparison_path": metadata["comparison_path"],
        "validation_metrics": metadata["validation_metrics"],
        "test_metrics": metadata["test_metrics"],
    }


def _resolve_relative_path(*, base_path: Path, candidate_path: Path) -> Path:
    if candidate_path.is_absolute():
        return candidate_path.resolve()
    return (base_path / candidate_path).resolve()


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()
