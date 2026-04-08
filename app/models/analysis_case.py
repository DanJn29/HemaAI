from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.models.enums import AnalysisSourceType


class AnalysisCase(TimestampMixin, Base):
    __tablename__ = "analysis_cases"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_code: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    sex: Mapped[str] = mapped_column(String(16), nullable=False)
    age: Mapped[int] = mapped_column(nullable=False)
    source_type: Mapped[AnalysisSourceType] = mapped_column(
        String(32),
        nullable=False,
        default=AnalysisSourceType.MANUAL.value,
        server_default=AnalysisSourceType.MANUAL.value,
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    values = relationship(
        "AnalysisValue",
        back_populates="analysis_case",
        cascade="all, delete-orphan",
        order_by="AnalysisValue.id",
    )
    results = relationship(
        "AnalysisResult",
        back_populates="analysis_case",
        cascade="all, delete-orphan",
        order_by="AnalysisResult.rank_position",
    )

