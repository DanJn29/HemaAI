from datetime import datetime
from decimal import Decimal

from app.schemas.base import AppBaseModel


class IndicatorResponse(AppBaseModel):
    id: int
    code: str
    name: str
    unit: str
    description: str | None
    created_at: datetime


class IndicatorMetadataResponse(AppBaseModel):
    id: int
    code: str
    name: str
    unit: str
    description: str | None
    normal_min: Decimal | None = None
    normal_max: Decimal | None = None
    min_allowed: Decimal
    max_allowed: Decimal
    warning_low: Decimal | None = None
    warning_high: Decimal | None = None
