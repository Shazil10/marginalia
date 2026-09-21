"""Environment-backed settings for Agent 2."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="openai/gpt-4o-mini", alias="OPENROUTER_MODEL")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    openrouter_app_name: str = Field(default="quantbros-agent2", alias="OPENROUTER_APP_NAME")
    openrouter_site_url: str | None = Field(default=None, alias="OPENROUTER_SITE_URL")
    openrouter_timeout_seconds: float = Field(default=60.0, alias="OPENROUTER_TIMEOUT_SECONDS")
    openrouter_max_retries: int = Field(default=2, alias="OPENROUTER_MAX_RETRIES")
    openrouter_retry_backoff_seconds: float = Field(
        default=1.5,
        alias="OPENROUTER_RETRY_BACKOFF_SECONDS",
    )

    papers_dir: str = Field(default="Papers", alias="AGENT2_PAPERS_DIR")
    database_path: str = Field(default="data/papers.sqlite", alias="AGENT2_DB_PATH")
    fixtures_dir: str = Field(default="fixtures/papers", alias="AGENT2_FIXTURES_DIR")
    artifacts_dir: str = Field(default="artifacts", alias="AGENT2_ARTIFACTS_DIR")


@lru_cache
def get_settings() -> Settings:
    return Settings()
