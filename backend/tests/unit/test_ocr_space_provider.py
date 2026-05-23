import httpx
import pytest

from app.core.exceptions import DomainValidationError, UpstreamServiceError, UpstreamTimeoutError
from app.services.ocr.ocr_space_provider import OcrSpaceProvider


def test_ocr_space_provider_returns_parsed_text_from_success_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.test/parse"
        body = request.content.decode("utf-8", errors="ignore")
        assert "language" in body
        assert "eng" in body
        assert "OCREngine" in body
        assert "2" in body
        return httpx.Response(
            200,
            json={
                "IsErroredOnProcessing": False,
                "ParsedResults": [{"ParsedText": "HGB 142\nHCT 42 %"}],
            },
        )

    provider = OcrSpaceProvider(
        api_key="test-key",
        endpoint="https://example.test/parse",
        engine=2,
        timeout_seconds=30,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert provider.extract_text(file_bytes=b"image", filename="cbc.png", content_type="image/png") == "HGB 142\nHCT 42 %"


def test_ocr_space_provider_raises_validation_error_for_ocr_space_error() -> None:
    provider = OcrSpaceProvider(
        api_key="test-key",
        endpoint="https://example.test/parse",
        engine=2,
        timeout_seconds=30,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={"IsErroredOnProcessing": True, "ErrorMessage": ["Invalid image file"]},
                )
            )
        ),
    )

    with pytest.raises(DomainValidationError, match="Invalid image file"):
        provider.extract_text(file_bytes=b"image", filename="cbc.png", content_type="image/png")


def test_ocr_space_provider_raises_validation_error_for_missing_parsed_text() -> None:
    provider = OcrSpaceProvider(
        api_key="test-key",
        endpoint="https://example.test/parse",
        engine=2,
        timeout_seconds=30,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json={"IsErroredOnProcessing": False, "ParsedResults": []})
            )
        ),
    )

    with pytest.raises(DomainValidationError, match="no parsed text"):
        provider.extract_text(file_bytes=b"image", filename="cbc.png", content_type="image/png")


def test_ocr_space_provider_maps_timeout_to_gateway_timeout() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout")

    provider = OcrSpaceProvider(
        api_key="test-key",
        endpoint="https://example.test/parse",
        engine=2,
        timeout_seconds=30,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(UpstreamTimeoutError):
        provider.extract_text(file_bytes=b"image", filename="cbc.png", content_type="image/png")


def test_ocr_space_provider_maps_server_error_to_bad_gateway() -> None:
    provider = OcrSpaceProvider(
        api_key="test-key",
        endpoint="https://example.test/parse",
        engine=2,
        timeout_seconds=30,
        client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500, text="bad"))),
    )

    with pytest.raises(UpstreamServiceError):
        provider.extract_text(file_bytes=b"image", filename="cbc.png", content_type="image/png")
