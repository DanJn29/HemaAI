from datetime import datetime

from app.schemas.base import AppBaseModel


class DiseaseResponse(AppBaseModel):
    id: int
    code: str
    name: str
    category: str
    description: str
    severity_level: str
    is_active: bool
    created_at: datetime

