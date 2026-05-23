from typing import Protocol


class OcrProvider(Protocol):
    def extract_text(self, *, file_bytes: bytes, filename: str, content_type: str) -> str:
        """Return raw OCR text extracted from an uploaded file."""
