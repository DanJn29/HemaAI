from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class DeviationState(Base):
    __tablename__ = "deviation_states"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    severity_rank: Mapped[int] = mapped_column(Integer, nullable=False)

    indicator_rules = relationship("IndicatorRule", back_populates="deviation_state")
    analysis_values = relationship("AnalysisValue", back_populates="deviation_state")
    pattern_rule_conditions = relationship("PatternRuleCondition", back_populates="deviation_state")

