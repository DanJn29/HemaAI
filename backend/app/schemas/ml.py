from app.schemas.base import AppBaseModel


class PredictionModelInfoResponse(AppBaseModel):
    model_name: str
    dataset_variant: str
    feature_mode: str
    include_rule_scores: bool


class ModelMetricsResponse(AppBaseModel):
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    top3_accuracy: float


class ModelInfoResponse(PredictionModelInfoResponse):
    selection_rule: str
    artifact_format_version: int
    model_path: str
    comparison_path: str
    validation_metrics: ModelMetricsResponse
    test_metrics: ModelMetricsResponse


class PredictionScoreResponse(AppBaseModel):
    label: str
    probability: float


class MLPredictResponse(AppBaseModel):
    predicted_label: str
    top_3_predictions: list[PredictionScoreResponse]
    model_info: PredictionModelInfoResponse


class RuleEngineComparisonResponse(AppBaseModel):
    top1_label: str
    top3_labels: list[str]


class MLPredictCompareResponse(MLPredictResponse):
    rule_engine: RuleEngineComparisonResponse
