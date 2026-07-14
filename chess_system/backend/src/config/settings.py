"""
Centralized environment configuration.

Loads variables from `.env` (local dev) or from Railway's injected
environment variables (production) via pydantic-settings. All other
modules should import `settings` from here rather than calling
`os.getenv` directly, so there is a single source of truth for config.
"""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, populated from environment variables."""

    environment: str = "local"

    # Comma-separated origins in .env are split into a list here.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_checkpoint_path: str = "./checkpoints/fischer_bc_v1.pt"
    stockfish_path: str = "/usr/games/stockfish"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def cors_origins_list(self) -> List[str]:
        """Return CORS_ORIGINS as a clean list of strings for FastAPI's middleware."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so the .env file is only parsed once per process."""
    return Settings()


settings = get_settings()
