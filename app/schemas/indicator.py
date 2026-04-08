from datetime import datetime

from app.schemas.base import AppBaseModel


class IndicatorResponse(AppBaseModel):
    id: int
    code: str
    name: str
    unit: str
    description: str | None
    created_at: datetime

