from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import DomainValidationError
from app.models.reference_range import ReferenceRange


class ReferenceRangeService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_for_indicator(self, indicator_id: int, sex: str, age: int) -> ReferenceRange:
        stmt = (
            select(ReferenceRange)
            .where(ReferenceRange.indicator_id == indicator_id)
            .where(ReferenceRange.sex == sex)
            .where(ReferenceRange.age_min <= age)
            .where(ReferenceRange.age_max >= age)
            .order_by(ReferenceRange.age_min.desc())
        )
        reference_range = self.session.scalar(stmt)
        if reference_range is None:
            raise DomainValidationError(
                f"No reference range configured for indicator_id={indicator_id}, sex={sex}, age={age}."
            )
        return reference_range

