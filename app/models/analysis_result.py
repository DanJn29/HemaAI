from decimal import Decimal

from sqlalchemy import ForeignKey, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.models.enums import AnalysisResultSource


class AnalysisResult(TimestampMixin, Base):
    __tablename__ = "analysis_results"
    __table_args__ = (
        UniqueConstraint("analysis_case_id", "disease_id", name="uq_analysis_result_case_disease"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_case_id: Mapped[int] = mapped_column(ForeignKey("analysis_cases.id"), nullable=False, index=True)
    disease_id: Mapped[int] = mapped_column(ForeignKey("diseases.id"), nullable=False, index=True)
    total_score: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    rank_position: Mapped[int] = mapped_column(nullable=False, index=True)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(10, 3), nullable=True)
    result_source: Mapped[AnalysisResultSource] = mapped_column(
        String(32),
        nullable=False,
        default=AnalysisResultSource.RULE_ENGINE.value,
        server_default=AnalysisResultSource.RULE_ENGINE.value,
    )

    analysis_case = relationship("AnalysisCase", back_populates="results")
    disease = relationship("Disease", back_populates="analysis_results")
    explanations = relationship(
        "AnalysisResultExplanation",
        back_populates="analysis_result",
        cascade="all, delete-orphan",
        order_by="AnalysisResultExplanation.id",
    )

