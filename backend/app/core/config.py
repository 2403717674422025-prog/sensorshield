"""
core/config.py
==============
Central configuration loaded from environment variables.
Using pydantic-settings so every value is typed and validated at startup.
If a required variable is missing, the app refuses to start with a clear error
rather than failing silently at runtime.
"""

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    APP_ENV: str = "development"
    APP_VERSION: str = "0.1.0"
    SECRET_KEY: str = "change-this-in-production"

    # ------------------------------------------------------------------ #
    # API Server
    # ------------------------------------------------------------------ #
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = (
        "postgresql+asyncpg://sensorshield:sensorshield@localhost:5432/sensorshield"
    )
    DATABASE_URL_SYNC: str = (
        "postgresql://sensorshield:sensorshield@localhost:5432/sensorshield"
    )

    # ------------------------------------------------------------------ #
    # MQTT
    # ------------------------------------------------------------------ #
    MQTT_HOST: str = "localhost"
    MQTT_PORT: int = 1883
    MQTT_TOPIC_PREFIX: str = "sensorshield"

    # ------------------------------------------------------------------ #
    # ML
    # ------------------------------------------------------------------ #
    MODEL_DIR: str = "models"
    RANDOM_SEED: int = 42
    ANOMALY_THRESHOLD: float = 0.85
    TRUST_HIGH_THRESHOLD: float = 80.0
    TRUST_LOW_THRESHOLD: float = 50.0

    # ------------------------------------------------------------------ #
    # CORS
    # ------------------------------------------------------------------ #
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------ #
    # Pydantic-settings config
    # ------------------------------------------------------------------ #
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("APP_ENV")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"APP_ENV must be one of {allowed}")
        return v


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    Use FastAPI's Depends(get_settings) to inject config into routes.
    The lru_cache means .env is read only once per process lifecycle.
    """
    return Settings()


# Module-level singleton for non-DI use (ML modules, scripts, etc.)
settings = get_settings()
