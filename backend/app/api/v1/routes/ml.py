from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.ml.inference.service import MLInferenceService
from app.schemas.analysis import AnalysisCreateRequest
from app.schemas.ml import MLPredictCompareResponse, MLPredictResponse, ModelInfoResponse

router = APIRouter(prefix="/ml")


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info(
    session: Session = Depends(get_db),
) -> ModelInfoResponse:
    service = MLInferenceService(session)
    return ModelInfoResponse.model_validate(service.get_model_info())


@router.post("/predict", response_model=MLPredictResponse)
def predict(
    payload: AnalysisCreateRequest,
    session: Session = Depends(get_db),
) -> MLPredictResponse:
    service = MLInferenceService(session)
    return MLPredictResponse.model_validate(service.predict(payload))


@router.post("/predict-and-compare", response_model=MLPredictCompareResponse)
def predict_and_compare(
    payload: AnalysisCreateRequest,
    session: Session = Depends(get_db),
) -> MLPredictCompareResponse:
    service = MLInferenceService(session)
    return MLPredictCompareResponse.model_validate(service.predict_and_compare(payload))
