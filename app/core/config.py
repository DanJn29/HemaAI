from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "HemaAI"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    database_url_override: str | None = Field(default=None, validation_alias="DATABASE_URL", repr=False)
    test_database_url_override: str | None = Field(
        default=None,
        validation_alias="TEST_DATABASE_URL",
        repr=False,
    )
    db_host: str | None = Field(default=None, validation_alias="DB_HOST")
    db_port: int | None = Field(default=None, validation_alias="DB_PORT")
    db_name: str | None = Field(default=None, validation_alias="DB_NAME")
    db_user: str | None = Field(default=None, validation_alias="DB_USER")
    db_password: str | None = Field(default=None, validation_alias="DB_PASSWORD", repr=False)
    test_db_host: str | None = Field(default=None, validation_alias="TEST_DB_HOST")
    test_db_port: int | None = Field(default=None, validation_alias="TEST_DB_PORT")
    test_db_name: str | None = Field(default=None, validation_alias="TEST_DB_NAME")
    test_db_user: str | None = Field(default=None, validation_alias="TEST_DB_USER")
    test_db_password: str = Field(
        default=None,
        validation_alias="TEST_DB_PASSWORD",
        repr=False,
    )
    min_pathology_score: float = Field(default=3.0)
    strong_pattern_bonus_threshold: float = Field(default=3.0)
    min_persisted_score: float = Field(default=1.0)
    max_persisted_non_normal: int = Field(default=5)
    max_returned_hypotheses: int = Field(default=3)
    analysis_disclaimer: str = (
        "This system provides ranked hypotheses for educational "
        "decision-support purposes and is not a definitive medical diagnosis."
    )

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        self._assert_database_parts(
            prefix="DB",
            values={
                "DB_HOST": self.db_host,
                "DB_PORT": self.db_port,
                "DB_NAME": self.db_name,
                "DB_USER": self.db_user,
                "DB_PASSWORD": self.db_password,
            },
        )
        return self._build_database_url(
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
            username=self.db_user,
            password=self.db_password,
        )

    @property
    def test_database_url(self) -> str:
        if self.test_database_url_override:
            return self.test_database_url_override
        self._assert_database_parts(
            prefix="TEST_DB",
            values={
                "TEST_DB_HOST": self.test_db_host,
                "TEST_DB_PORT": self.test_db_port,
                "TEST_DB_NAME": self.test_db_name,
                "TEST_DB_USER": self.test_db_user,
                "TEST_DB_PASSWORD": self.test_db_password,
            },
        )
        return self._build_database_url(
            host=self.test_db_host,
            port=self.test_db_port,
            database=self.test_db_name,
            username=self.test_db_user,
            password=self.test_db_password,
        )

    @staticmethod
    def _build_database_url(
        *,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
    ) -> str:
        return URL.create(
            drivername="postgresql+psycopg",
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
        ).render_as_string(hide_password=False)

    @staticmethod
    def _assert_database_parts(*, prefix: str, values: dict[str, object | None]) -> None:
        missing = [key for key, value in values.items() if value in (None, "")]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"Missing required database environment variables for {prefix}: {joined}"
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
