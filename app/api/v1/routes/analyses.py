from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.schemas.analysis import AnalysisCreateRequest, AnalysisResponse
from app.services.analysis_orchestrator import AnalysisOrchestrator

router = APIRouter()


@router.post("/analyses", response_model=AnalysisResponse)
def create_analysis(
    payload: AnalysisCreateRequest,
    session: Session = Depends(get_db),
) -> AnalysisResponse:
    orchestrator = AnalysisOrchestrator(session)
    return orchestrator.create_analysis(payload)


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(
    analysis_id: int,
    session: Session = Depends(get_db),
) -> AnalysisResponse:
    orchestrator = AnalysisOrchestrator(session)
    return orchestrator.get_analysis(analysis_id)


@router.post("/recompute/{analysis_id}", response_model=AnalysisResponse)
def recompute_analysis(
    analysis_id: int,
    session: Session = Depends(get_db),
) -> AnalysisResponse:
    orchestrator = AnalysisOrchestrator(session)
    return orchestrator.recompute_analysis(analysis_id)
