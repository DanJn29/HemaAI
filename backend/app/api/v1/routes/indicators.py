from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.exceptions import DomainValidationError
from app.core.indicator_input_limits import get_indicator_input_limit
from app.repositories.catalog_repository import CatalogRepository
from app.schemas.analysis import ALLOWED_SEXES
from app.schemas.indicator import IndicatorMetadataResponse, IndicatorResponse
from app.services.reference_range_service import ReferenceRangeService

router = APIRouter(prefix="/indicators")


@router.get("/metadata", response_model=list[IndicatorMetadataResponse])
def list_indicator_metadata(
    sex: str | None = Query(default=None),
    age: int | None = Query(default=None),
    session: Session = Depends(get_db),
) -> list[IndicatorMetadataResponse]:
    normalized_sex = _normalize_optional_sex(sex)
    _validate_metadata_scope(normalized_sex, age)

    repository = CatalogRepository(session)
    reference_range_service = ReferenceRangeService(session)
    metadata: list[IndicatorMetadataResponse] = []

    for indicator in repository.list_indicators():
        input_limit = get_indicator_input_limit(indicator.code)
        if input_limit is None:
            continue

        normal_min = None
        normal_max = None
        if normalized_sex is not None and age is not None:
            reference_range = reference_range_service.get_for_indicator(
                indicator_id=indicator.id,
                sex=normalized_sex,
                age=age,
            )
            normal_min = reference_range.normal_min
            normal_max = reference_range.normal_max

        metadata.append(
            IndicatorMetadataResponse(
                id=indicator.id,
                code=indicator.code,
                name=indicator.name,
                unit=indicator.unit,
                description=indicator.description,
                normal_min=normal_min,
                normal_max=normal_max,
                min_allowed=input_limit.min_allowed,
                max_allowed=input_limit.max_allowed,
                warning_low=input_limit.warning_low,
                warning_high=input_limit.warning_high,
            )
        )

    return metadata


@router.get("", response_model=list[IndicatorResponse])
def list_indicators(session: Session = Depends(get_db)) -> list[IndicatorResponse]:
    repository = CatalogRepository(session)
    return [IndicatorResponse.model_validate(indicator) for indicator in repository.list_indicators()]


def _normalize_optional_sex(sex: str | None) -> str | None:
    if sex is None:
        return None
    normalized = sex.strip().lower()
    if normalized not in ALLOWED_SEXES:
        raise DomainValidationError("sex must be one of: male, female")
    return normalized


def _validate_metadata_scope(sex: str | None, age: int | None) -> None:
    if (sex is None) != (age is None):
        raise DomainValidationError("sex and age must be provided together for normal ranges.")
    if age is not None and (age < 18 or age > 120):
        raise DomainValidationError("age must be between 18 and 120 for this MVP")
