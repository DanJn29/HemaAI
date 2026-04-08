import os
from collections.abc import Generator

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy.engine import URL
from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import Session

if os.getenv("TEST_DATABASE_URL"):
    os.environ.setdefault("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
elif all(os.getenv(key) for key in ["TEST_DB_HOST", "TEST_DB_PORT", "TEST_DB_NAME", "TEST_DB_USER", "TEST_DB_PASSWORD"]):
    os.environ.setdefault(
        "DATABASE_URL",
        URL.create(
            drivername="postgresql+psycopg",
            host=os.environ["TEST_DB_HOST"],
            port=int(os.environ["TEST_DB_PORT"]),
            database=os.environ["TEST_DB_NAME"],
            username=os.environ["TEST_DB_USER"],
            password=os.environ["TEST_DB_PASSWORD"],
        ).render_as_string(hide_password=False),
    )
else:
    missing = [
        key
        for key in ["TEST_DATABASE_URL", "TEST_DB_HOST", "TEST_DB_PORT", "TEST_DB_NAME", "TEST_DB_USER", "TEST_DB_PASSWORD"]
        if not os.getenv(key)
    ]
    raise RuntimeError(
        "Missing test database configuration. Set TEST_DATABASE_URL or all TEST_DB_* variables "
        f"in your local .env before running tests. Missing candidates: {', '.join(missing)}"
    )

os.environ.setdefault("TEST_DATABASE_URL", os.environ["DATABASE_URL"])

from app.core.config import get_settings
from app.core.db import get_session_factory, initialize_engine
from app.main import create_app
from app.models.analysis_case import AnalysisCase
from app.models.analysis_result import AnalysisResult
from app.models.analysis_result_explanation import AnalysisResultExplanation
from app.models.analysis_value import AnalysisValue
from app.seed.seed_data import seed_database


@pytest.fixture(scope="session", autouse=True)
def prepared_database() -> Generator[None, None, None]:
    get_settings.cache_clear()
    settings = get_settings()
    initialize_engine(settings.database_url)

    reset_engine = create_engine(settings.database_url, isolation_level="AUTOCOMMIT")
    with reset_engine.connect() as connection:
        connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    reset_engine.dispose()

    alembic_config = Config("alembic.ini")
    alembic_config.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(alembic_config, "head")

    session_factory = get_session_factory()
    with session_factory() as session:
        with session.begin():
            seed_database(session)
    yield


@pytest.fixture(autouse=True)
def clean_analysis_tables() -> Generator[None, None, None]:
    session_factory = get_session_factory()
    with session_factory() as session:
        with session.begin():
            session.execute(delete(AnalysisResultExplanation))
            session.execute(delete(AnalysisResult))
            session.execute(delete(AnalysisValue))
            session.execute(delete(AnalysisCase))
    yield


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    session_factory = get_session_factory()
    with session_factory() as session:
        yield session


@pytest.fixture
def client(prepared_database: None) -> Generator[TestClient, None, None]:
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
