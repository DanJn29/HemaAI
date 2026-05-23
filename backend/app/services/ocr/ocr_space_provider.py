from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from app.core.exceptions import DomainValidationError, UpstreamServiceError, UpstreamTimeoutError


class OcrSpaceProvider:
    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        engine: int,
        timeout_seconds: int,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.engine = engine
        self.timeout_seconds = timeout_seconds
        self.client = client

    def extract_text(self, *, file_bytes: bytes, filename: str, content_type: str) -> str:
        data = {
            "apikey": self.api_key,
            "language": "eng",
            "isOverlayRequired": "false",
            "isTable": "true",
            "scale": "true",
            "detectOrientation": "true",
            "OCREngine": str(self.engine),
        }
        files = {"file": (filename, file_bytes, content_type)}

        try:
            if self.client is not None:
                response = self.client.post(self.endpoint, data=data, files=files, timeout=self.timeout_seconds)
            else:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(self.endpoint, data=data, files=files)
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError("OCR.space request timed out. Please try again later.") from exc
        except httpx.HTTPError as exc:
            raise UpstreamServiceError("OCR.space request failed. Please try again later.") from exc

        if response.status_code >= 500:
            raise UpstreamServiceError("OCR.space is temporarily unavailable. Please try again later.")
        if response.status_code >= 400:
            raise DomainValidationError(f"OCR.space rejected the upload with status {response.status_code}.")

        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamServiceError("OCR.space returned an unreadable response.") from exc

        return _extract_parsed_text(payload)


def _extract_parsed_text(payload: Mapping[str, Any]) -> str:
    if bool(payload.get("IsErroredOnProcessing")):
        message = _extract_error_message(payload)
        raise DomainValidationError(message)

    parsed_results = payload.get("ParsedResults")
    if not isinstance(parsed_results, list):
        raise DomainValidationError("OCR.space returned no parsed text.")

    parsed_texts: list[str] = []
    for result in parsed_results:
        if isinstance(result, Mapping):
            parsed_text = result.get("ParsedText")
            if isinstance(parsed_text, str) and parsed_text.strip():
                parsed_texts.append(parsed_text.strip())

    if not parsed_texts:
        raise DomainValidationError("OCR.space returned no parsed text.")

    return "\n".join(parsed_texts)


def _extract_error_message(payload: Mapping[str, Any]) -> str:
    for key in ("ErrorMessage", "ErrorDetails"):
        value = payload.get(key)
        if isinstance(value, list):
            joined = " ".join(str(item) for item in value if item)
            if joined:
                return joined
        if isinstance(value, str) and value:
            return value
    return "OCR.space could not process this image."
