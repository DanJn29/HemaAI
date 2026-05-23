from sqlalchemy import CheckConstraint, Enum, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.enums import DeviationFamily, PatternMatchMode, enum_values


class PatternRuleCondition(Base):
    __tablename__ = "pattern_rule_conditions"
    __table_args__ = (
        CheckConstraint(
            "("
            "match_mode = 'exact' AND deviation_state_id IS NOT NULL AND deviation_family IS NULL"
            ") OR ("
            "match_mode = 'family' AND deviation_state_id IS NULL AND deviation_family IS NOT NULL"
            ")",
            name="ck_pattern_rule_condition_match_mode",
        ),
        Index("ix_pattern_rule_conditions_rule_id", "pattern_rule_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    pattern_rule_id: Mapped[int] = mapped_column(ForeignKey("pattern_rules.id"), nullable=False)
    indicator_id: Mapped[int] = mapped_column(ForeignKey("indicators.id"), nullable=False)
    deviation_state_id: Mapped[int | None] = mapped_column(
        ForeignKey("deviation_states.id"),
        nullable=True,
    )
    match_mode: Mapped[PatternMatchMode] = mapped_column(
        Enum(
            PatternMatchMode,
            native_enum=False,
            values_callable=enum_values,
        ),
        nullable=False,
        default=PatternMatchMode.EXACT,
    )
    deviation_family: Mapped[DeviationFamily | None] = mapped_column(
        Enum(
            DeviationFamily,
            native_enum=False,
            values_callable=enum_values,
        ),
        nullable=True,
    )

    pattern_rule = relationship("PatternRule", back_populates="conditions")
    indicator = relationship("Indicator", back_populates="pattern_rule_conditions")
    deviation_state = relationship("DeviationState", back_populates="pattern_rule_conditions")
