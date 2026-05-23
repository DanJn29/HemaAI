from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from starlette.requests import Request
from typing import Any


class AppError(Exception):
    """Base class for application-level exceptions."""

    def __init__(self, detail: Any, status_code: int) -> None:
        self.detail = detail
        self.status_code = status_code
        super().__init__(str(detail))


class NotFoundError(AppError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail, status_code=status.HTTP_404_NOT_FOUND)


class DomainValidationError(AppError):
    def __init__(self, detail: Any) -> None:
        super().__init__(detail=detail, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


class UpstreamServiceError(AppError):
    def __init__(self, detail: Any) -> None:
        super().__init__(detail=detail, status_code=status.HTTP_502_BAD_GATEWAY)


class UpstreamTimeoutError(AppError):
    def __init__(self, detail: Any) -> None:
        super().__init__(detail=detail, status_code=status.HTTP_504_GATEWAY_TIMEOUT)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
