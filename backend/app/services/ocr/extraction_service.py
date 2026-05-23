from app.schemas.ocr import CBCOcrExtractResponse
from app.services.ocr.cbc_parser import parse_cbc_text
from app.services.ocr.provider import OcrProvider


class CBCOcrExtractionService:
    def __init__(self, provider: OcrProvider) -> None:
        self.provider = provider

    def extract(self, *, image_bytes: bytes, filename: str, content_type: str) -> CBCOcrExtractResponse:
        raw_text = self.provider.extract_text(
            file_bytes=image_bytes,
            filename=filename,
            content_type=content_type,
        )
        parsed = parse_cbc_text(raw_text)
        return CBCOcrExtractResponse(
            extracted_values=parsed.extracted_values,
            patient={
                "age": parsed.patient.age,
                "sex": parsed.patient.sex,
            },
            raw_text=parsed.raw_text,
            warnings=parsed.warnings,
        )
