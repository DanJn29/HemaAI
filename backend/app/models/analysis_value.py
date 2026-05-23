from decimal import Decimal

from sqlalchemy import ForeignKey, Numeric, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class AnalysisValue(TimestampMixin, Base):
    __tablename__ = "analysis_values"
    __table_args__ = (
        UniqueConstraint("analysis_case_id", "indicator_id", name="uq_analysis_case_indicator"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_case_id: Mapped[int] = mapped_column(ForeignKey("analysis_cases.id"), nullable=False, index=True)
    indicator_id: Mapped[int] = mapped_column(ForeignKey("indicators.id"), nullable=False, index=True)
    raw_value: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    deviation_state_id: Mapped[int | None] = mapped_column(
        ForeignKey("deviation_states.id"),
        nullable=True,
        index=True,
    )
    normalized_score: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)

    analysis_case = relationship("AnalysisCase", back_populates="values")
    indicator = relationship("Indicator", back_populates="analysis_values")
    deviation_state = relationship("DeviationState", back_populates="analysis_values")

