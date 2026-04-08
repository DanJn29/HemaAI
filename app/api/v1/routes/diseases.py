from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.repositories.catalog_repository import CatalogRepository
from app.schemas.disease import DiseaseResponse

router = APIRouter(prefix="/diseases")


@router.get("", response_model=list[DiseaseResponse])
def list_diseases(session: Session = Depends(get_db)) -> list[DiseaseResponse]:
    repository = CatalogRepository(session)
    return [DiseaseResponse.model_validate(disease) for disease in repository.list_active_diseases()]

