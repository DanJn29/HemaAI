from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Indicator(TimestampMixin, Base):
    __tablename__ = "indicators"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    unit: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    reference_ranges = relationship("ReferenceRange", back_populates="indicator")
    indicator_rules = relationship("IndicatorRule", back_populates="indicator")
    analysis_values = relationship("AnalysisValue", back_populates="indicator")
    pattern_rule_conditions = relationship("PatternRuleCondition", back_populates="indicator")

