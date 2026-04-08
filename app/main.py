from fastapi import FastAPI

from app.api.router import router as api_router
from app.core.config import get_settings
from app.core.db import initialize_engine
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()
    initialize_engine(settings.database_url)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Rule-based CBC interpretation API that returns ranked clinical hypotheses "
            "for decision-support and educational use."
        ),
    )
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
