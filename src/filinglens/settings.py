"""Application settings, loaded from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from filinglens.edgar.useragent import validate_user_agent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    sec_user_agent: str = Field(..., description="App name plus contact email, per SEC policy.")
    database_url: str | None = None
    cache_dir: Path = Path(".cache/sec")
    sec_max_requests_per_second: float = Field(8.0, gt=0, le=10)
    anthropic_api_key: str | None = None

    # Agent (Phase 2+). Model names live in config, not code.
    agent_model: str = "claude-opus-5-5"
    agent_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    agent_max_turns: int = Field(12, ge=1, le=50)
    agent_max_budget_usd: float = Field(1.0, gt=0, le=20)

    # Risk-factor classification (Phase 3).
    classifier_model: str = "claude-opus-5-5"
    classifier_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    classifier_factors_per_request: int = Field(8, ge=1, le=25)

    # LLM-as-judge for the eval suite (Phase 4).
    judge_model: str = "claude-opus-5-5"
    judge_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"

    # Web API (Phase 6). Set API_TOKEN anywhere but local development.
    api_token: str | None = None
    api_cors_origins: list[str] = ["http://localhost:3000"]

    # Stock prices (free key from https://www.alphavantage.co/support/#api-key).
    alphavantage_api_key: str | None = None

    @field_validator("sec_user_agent")
    @classmethod
    def _check_user_agent(cls, value: str) -> str:
        return validate_user_agent(value)


class DatabaseSettings(BaseSettings):
    """Just the database URL, so migrations don't require SEC settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # values come from the environment


def get_database_url() -> str:
    """The SQLAlchemy URL, with `local` resolved to the embedded dev server (Alembic needs it)."""
    from filinglens.db.localdb import resolve

    return resolve(DatabaseSettings().database_url)
