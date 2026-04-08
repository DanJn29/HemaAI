from decimal import Decimal

from sqlalchemy import ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class PatternRule(TimestampMixin, Base):
    __tablename__ = "pattern_rules"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    disease_id: Mapped[int] = mapped_column(ForeignKey("diseases.id"), nullable=False, index=True)
    bonus_weight: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    rule_description: Mapped[str] = mapped_column(Text, nullable=False)

    disease = relationship("Disease", back_populates="pattern_rules")
    conditions = relationship(
        "PatternRuleCondition",
        back_populates="pattern_rule",
        cascade="all, delete-orphan",
        order_by="PatternRuleCondition.id",
    )

