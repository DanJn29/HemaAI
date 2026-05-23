from app.core.config import get_settings
from app.services.ocr.ocr_space_provider import OcrSpaceProvider


def test_ocr_endpoint_rejects_unsupported_file_type(client) -> None:
    response = client.post(
        "/api/v1/ocr/cbc-extract",
        files={"file": ("cbc.txt", b"WBC 6.2", "text/plain")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Unsupported file type. Upload a jpg, jpeg, or png image."


def test_ocr_endpoint_rejects_missing_api_key(client, monkeypatch) -> None:
    monkeypatch.setenv("OCR_SPACE_API_KEY", "")
    get_settings.cache_clear()

    response = client.post(
        "/api/v1/ocr/cbc-extract",
        files={"file": ("cbc.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "OCR_SPACE_API_KEY is not configured."


def test_ocr_endpoint_uses_provider_and_returns_parsed_values(client, monkeypatch) -> None:
    monkeypatch.setenv("OCR_SPACE_API_KEY", "test-key")
    get_settings.cache_clear()

    def fake_extract_text(self, *, file_bytes: bytes, filename: str, content_type: str) -> str:
        assert file_bytes == b"fake-image"
        assert filename == "cbc.png"
        assert content_type == "image/png"
        return "Hemoglobin 14.2 g/dL\nHCT 42 %\nPLT 62"

    monkeypatch.setattr(OcrSpaceProvider, "extract_text", fake_extract_text)

    response = client.post(
        "/api/v1/ocr/cbc-extract",
        files={"file": ("cbc.png", b"fake-image", "image/png")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["extracted_values"]["HGB"] == 142
    assert payload["extracted_values"]["HCT"] == 0.42
    assert payload["extracted_values"]["PLT"] == 62
    assert payload["patient"] == {"age": None, "sex": None}
    assert "confidence" not in payload
