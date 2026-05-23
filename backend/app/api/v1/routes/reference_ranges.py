from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.repositories.catalog_repository import CatalogRepository
from app.schemas.reference_range import ReferenceRangeResponse

router = APIRouter(prefix="/reference-ranges")


@router.get("", response_model=list[ReferenceRangeResponse])
def list_reference_ranges(
    sex: str | None = Query(default=None),
    indicator_code: str | None = Query(default=None),
    session: Session = Depends(get_db),
) -> list[ReferenceRangeResponse]:
    repository = CatalogRepository(session)
    reference_ranges = repository.list_reference_ranges(
        sex=sex.lower() if sex else None,
        indicator_code=indicator_code.upper() if indicator_code else None,
    )
    return [
        ReferenceRangeResponse(
            id=reference_range.id,
            indicator_code=reference_range.indicator.code,
            indicator_name=reference_range.indicator.name,
            sex=reference_range.sex,
            age_min=reference_range.age_min,
            age_max=reference_range.age_max,
            normal_min=reference_range.normal_min,
            normal_max=reference_range.normal_max,
            mild_low_threshold=reference_range.mild_low_threshold,
            moderate_low_threshold=reference_range.moderate_low_threshold,
            severe_low_threshold=reference_range.severe_low_threshold,
            mild_high_threshold=reference_range.mild_high_threshold,
            moderate_high_threshold=reference_range.moderate_high_threshold,
            severe_high_threshold=reference_range.severe_high_threshold,
            created_at=reference_range.created_at,
        )
        for reference_range in reference_ranges
    ]

