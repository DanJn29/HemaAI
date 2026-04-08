from datetime import datetime
from decimal import Decimal

from app.schemas.base import AppBaseModel


class ReferenceRangeResponse(AppBaseModel):
    id: int
    indicator_code: str
    indicator_name: str
    sex: str
    age_min: int
    age_max: int
    normal_min: Decimal
    normal_max: Decimal
    mild_low_threshold: Decimal
    moderate_low_threshold: Decimal
    severe_low_threshold: Decimal
    mild_high_threshold: Decimal
    moderate_high_threshold: Decimal
    severe_high_threshold: Decimal
    created_at: datetime

