from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import Field, field_validator, model_validator

from app.schemas.base import AppBaseModel

ALLOWED_SEXES = {"male", "female"}


class IndicatorValueInput(AppBaseModel):
    indicator_code: str
    raw_value: Decimal

    @field_validator("indicator_code")
    @classmethod
    def normalize_indicator_code(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("indicator_code must not be empty")
        return value

    @field_validator("raw_value")
    @classmethod
    def validate_raw_value(cls, value: Decimal) -> Decimal:
        if value < 0:
            raise ValueError("raw_value must be greater than or equal to 0")
        return value


class AnalysisCreateRequest(AppBaseModel):
    sex: str
    age: int
    patient_code: str | None = None
    notes: str | None = None
    values: list[IndicatorValueInput] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_values_payload(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("values"), dict):
            payload = dict(data)
            payload["values"] = [
                {"indicator_code": code, "raw_value": raw_value}
                for code, raw_value in payload["values"].items()
            ]
            return payload
        return data

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_SEXES:
            raise ValueError("sex must be one of: male, female")
        return normalized

    @field_validator("age")
    @classmethod
    def validate_age(cls, value: int) -> int:
        if value < 18 or value > 120:
            raise ValueError("age must be between 18 and 120 for this MVP")
        return value

    @field_validator("values")
    @classmethod
    def validate_unique_indicators(
        cls,
        values: list[IndicatorValueInput],
    ) -> list[IndicatorValueInput]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for item in values:
            if item.indicator_code in seen:
                duplicates.add(item.indicator_code)
            seen.add(item.indicator_code)
        if duplicates:
            joined = ", ".join(sorted(duplicates))
            raise ValueError(f"duplicate indicator values submitted: {joined}")
        return values


class PatientSummary(AppBaseModel):
    sex: str
    age: int
    patient_code: str | None = None
    notes: str | None = None


class IndicatorInterpretationResponse(AppBaseModel):
    indicator_code: str
    indicator_name: str
    raw_value: Decimal
    unit: str
    deviation_state: str
    normalized_score: float | None


class ExplanationResponse(AppBaseModel):
    type: str
    text: str
    score_effect: float


class HypothesisResponse(AppBaseModel):
    rank: int
    disease_code: str
    disease_name: str
    total_score: float
    confidence: float | None
    explanations: list[ExplanationResponse]


class AnalysisResponse(AppBaseModel):
    analysis_id: int
    created_at: datetime
    patient: PatientSummary
    indicator_interpretation: list[IndicatorInterpretationResponse]
    top_hypotheses: list[HypothesisResponse]
    disclaimer: str

