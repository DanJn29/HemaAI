from decimal import Decimal

from sqlalchemy import Enum, ForeignKey, Numeric, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.models.enums import ExplanationSourceType, enum_values


class AnalysisResultExplanation(TimestampMixin, Base):
    __tablename__ = "analysis_result_explanations"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_result_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_type: Mapped[ExplanationSourceType] = mapped_column(
        Enum(
            ExplanationSourceType,
            native_enum=False,
            values_callable=enum_values,
        ),
        nullable=False,
    )
    source_id: Mapped[int | None] = mapped_column(nullable=True)
    explanation_text: Mapped[str] = mapped_column(Text, nullable=False)
    score_effect: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)

    analysis_result = relationship("AnalysisResult", back_populates="explanations")
