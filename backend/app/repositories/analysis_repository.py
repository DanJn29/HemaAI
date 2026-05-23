from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.models.analysis_case import AnalysisCase
from app.models.analysis_result import AnalysisResult
from app.models.analysis_value import AnalysisValue


class AnalysisRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_case(self, analysis_case: AnalysisCase) -> AnalysisCase:
        self.session.add(analysis_case)
        self.session.flush()
        return analysis_case

    def add_value(self, analysis_value: AnalysisValue) -> AnalysisValue:
        self.session.add(analysis_value)
        self.session.flush()
        return analysis_value

    def get_case(self, analysis_case_id: int) -> AnalysisCase | None:
        stmt = (
            select(AnalysisCase)
            .where(AnalysisCase.id == analysis_case_id)
            .execution_options(populate_existing=True)
            .options(
                selectinload(AnalysisCase.values).joinedload(AnalysisValue.indicator),
                selectinload(AnalysisCase.values).joinedload(AnalysisValue.deviation_state),
                selectinload(AnalysisCase.results)
                .joinedload(AnalysisResult.disease),
                selectinload(AnalysisCase.results)
                .selectinload(AnalysisResult.explanations),
            )
        )
        return self.session.scalar(stmt)

    def get_case_for_update(self, analysis_case_id: int) -> AnalysisCase | None:
        stmt = (
            select(AnalysisCase)
            .where(AnalysisCase.id == analysis_case_id)
            .execution_options(populate_existing=True)
            .options(
                selectinload(AnalysisCase.values).joinedload(AnalysisValue.indicator),
                selectinload(AnalysisCase.values).joinedload(AnalysisValue.deviation_state),
                selectinload(AnalysisCase.results).selectinload(AnalysisResult.explanations),
            )
        )
        return self.session.scalar(stmt)
