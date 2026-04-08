from decimal import Decimal

from sqlalchemy import Enum, ForeignKey, Numeric, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin
from app.models.enums import IndicatorRuleRelationType, enum_values


class IndicatorRule(TimestampMixin, Base):
    __tablename__ = "indicator_rules"
    __table_args__ = (
        UniqueConstraint(
            "indicator_id",
            "deviation_state_id",
            "disease_id",
            "relation_type",
            name="uq_indicator_rule_scope",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    indicator_id: Mapped[int] = mapped_column(ForeignKey("indicators.id"), nullable=False, index=True)
    deviation_state_id: Mapped[int] = mapped_column(
        ForeignKey("deviation_states.id"),
        nullable=False,
        index=True,
    )
    disease_id: Mapped[int] = mapped_column(ForeignKey("diseases.id"), nullable=False, index=True)
    relation_type: Mapped[IndicatorRuleRelationType] = mapped_column(
        Enum(
            IndicatorRuleRelationType,
            native_enum=False,
            values_callable=enum_values,
        ),
        nullable=False,
    )
    weight: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    evidence_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    indicator = relationship("Indicator", back_populates="indicator_rules")
    deviation_state = relationship("DeviationState", back_populates="indicator_rules")
    disease = relationship("Disease", back_populates="indicator_rules")
