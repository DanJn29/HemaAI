from collections.abc import Sequence

from sqlalchemy import Select, select
from sqlalchemy.orm import Session, joinedload

from app.models.deviation_state import DeviationState
from app.models.disease import Disease
from app.models.indicator import Indicator
from app.models.reference_range import ReferenceRange


class CatalogRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_indicators(self) -> Sequence[Indicator]:
        stmt = select(Indicator).order_by(Indicator.code)
        return self.session.scalars(stmt).all()

    def get_indicators_by_codes(self, codes: Sequence[str]) -> Sequence[Indicator]:
        stmt = select(Indicator).where(Indicator.code.in_(codes)).order_by(Indicator.code)
        return self.session.scalars(stmt).all()

    def list_active_diseases(self) -> Sequence[Disease]:
        stmt = select(Disease).where(Disease.is_active.is_(True)).order_by(Disease.name)
        return self.session.scalars(stmt).all()

    def get_disease_by_code(self, code: str) -> Disease | None:
        stmt = select(Disease).where(Disease.code == code)
        return self.session.scalar(stmt)

    def get_deviation_states(self) -> Sequence[DeviationState]:
        stmt = select(DeviationState).order_by(DeviationState.severity_rank)
        return self.session.scalars(stmt).all()

    def get_deviation_state_map(self) -> dict[str, DeviationState]:
        return {state.code: state for state in self.get_deviation_states()}

    def list_reference_ranges(
        self,
        sex: str | None = None,
        indicator_code: str | None = None,
    ) -> Sequence[ReferenceRange]:
        stmt: Select[tuple[ReferenceRange]] = (
            select(ReferenceRange)
            .options(joinedload(ReferenceRange.indicator))
            .order_by(Indicator.code, ReferenceRange.sex, ReferenceRange.age_min)
            .join(ReferenceRange.indicator)
        )
        if sex is not None:
            stmt = stmt.where(ReferenceRange.sex == sex)
        if indicator_code is not None:
            stmt = stmt.where(Indicator.code == indicator_code)
        return self.session.scalars(stmt).all()

