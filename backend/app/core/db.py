from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def initialize_engine(database_url: str | None = None) -> None:
    """Initialize or replace the SQLAlchemy engine/session factory."""

    global _engine, _session_factory

    if _engine is not None:
        _engine.dispose()

    settings = get_settings()
    target_url = database_url or settings.database_url
    _engine = create_engine(target_url, pool_pre_ping=True)
    _session_factory = sessionmaker(
        bind=_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )


def get_engine() -> Engine:
    if _engine is None:
        initialize_engine()
    assert _engine is not None
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    if _session_factory is None:
        initialize_engine()
    assert _session_factory is not None
    return _session_factory


def get_db() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()

