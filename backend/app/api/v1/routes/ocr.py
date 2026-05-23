from fastapi import APIRouter, Depends, Request
from starlette.datastructures import UploadFile

from app.core.config import Settings, get_settings
from app.core.exceptions import DomainValidationError
from app.schemas.ocr import CBCOcrExtractResponse
from app.services.ocr.extraction_service import CBCOcrExtractionService
from app.services.ocr.ocr_space_provider import OcrSpaceProvider

router = APIRouter(prefix="/ocr")

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}


@router.post(
    "/cbc-extract",
    response_model=CBCOcrExtractResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {"file": {"type": "string", "format": "binary"}},
                    }
                }
            },
        }
    },
)
async def extract_cbc_from_image(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> CBCOcrExtractResponse:
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        raise DomainValidationError("Request must be multipart/form-data with a file field.")

    try:
        form = await request.form()
    except (AssertionError, RuntimeError) as exc:
        raise DomainValidationError(
            "Could not parse multipart upload. Rebuild the API container so python-multipart is installed."
        ) from exc
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise DomainValidationError("Upload a CBC image using the 'file' form field.")

    if upload.content_type not in ALLOWED_IMAGE_TYPES:
        raise DomainValidationError("Unsupported file type. Upload a jpg, jpeg, or png image.")

    contents = await upload.read()
    if not contents:
        raise DomainValidationError("Uploaded file is empty.")
    max_file_size_bytes = settings.ocr_max_file_size_mb * 1024 * 1024
    if len(contents) > max_file_size_bytes:
        raise DomainValidationError(f"Uploaded file is too large. Maximum size is {settings.ocr_max_file_size_mb:g} MB.")

    if not settings.ocr_space_api_key:
        raise DomainValidationError("OCR_SPACE_API_KEY is not configured.")

    provider = OcrSpaceProvider(
        api_key=settings.ocr_space_api_key,
        endpoint=settings.ocr_space_endpoint,
        engine=settings.ocr_space_engine,
        timeout_seconds=settings.ocr_space_timeout_seconds,
    )
    return CBCOcrExtractionService(provider).extract(
        image_bytes=contents,
        filename=upload.filename or "cbc-upload",
        content_type=upload.content_type,
    )
