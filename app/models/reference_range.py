from decimal import Decimal

from sqlalchemy import ForeignKey, Index, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class ReferenceRange(TimestampMixin, Base):
    __tablename__ = "reference_ranges"
    __table_args__ = (
        UniqueConstraint("indicator_id", "sex", "age_min", "age_max", name="uq_reference_range_scope"),
        Index("ix_reference_ranges_lookup", "indicator_id", "sex", "age_min", "age_max"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    indicator_id: Mapped[int] = mapped_column(ForeignKey("indicators.id"), nullable=False)
    sex: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    age_min: Mapped[int] = mapped_column(nullable=False)
    age_max: Mapped[int] = mapped_column(nullable=False)
    normal_min: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    normal_max: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    mild_low_threshold: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    moderate_low_threshold: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    severe_low_threshold: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    mild_high_threshold: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    moderate_high_threshold: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    severe_high_threshold: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)

    indicator = relationship("Indicator", back_populates="reference_ranges")

