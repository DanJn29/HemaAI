from app.schemas.base import AppBaseModel


class OcrPatientResponse(AppBaseModel):
    age: int | None = None
    sex: str | None = None


class CBCOcrExtractResponse(AppBaseModel):
    extracted_values: dict[str, float | None]
    patient: OcrPatientResponse
    raw_text: str
    warnings: list[str]
