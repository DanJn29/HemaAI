from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.repositories.catalog_repository import CatalogRepository
from app.schemas.indicator import IndicatorResponse

router = APIRouter(prefix="/indicators")


@router.get("", response_model=list[IndicatorResponse])
def list_indicators(session: Session = Depends(get_db)) -> list[IndicatorResponse]:
    repository = CatalogRepository(session)
    return [IndicatorResponse.model_validate(indicator) for indicator in repository.list_indicators()]

