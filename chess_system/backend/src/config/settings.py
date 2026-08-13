"""
Centralized environment configuration.

Loads variables from `.env` (local dev) or from Railway's injected
environment variables (production) via pydantic-settings. All other
modules should import `settings` from here rather than calling
`os.getenv` directly, so there is a single source of truth for config.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, populated from environment variables."""

    environment: str = "local"

    # Comma-separated origins in .env are split into a list here.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_checkpoint_path: Path = Path("./checkpoints/best_fischer_bc.pth")
    stockfish_path: str = "/usr/games/stockfish"

    # Behavioral Cloning model architecture. These values are intentionally
    # centralized so both the trainer and inference service build identical
    # policies.
    model_input_channels: int = 18
    model_num_actions: int = 4672
    model_channels: int = 128
    model_residual_blocks: int = 8
    model_policy_channels: int = 32

    # Behavioral Cloning training. Every field may be overridden through the
    # corresponding upper-case environment variable, e.g. TRAINING_LEARNING_RATE.
    training_data_path: Path = Path("./data/cache/fischer_training_examples.jsonl")
    training_checkpoint_dir: Path = Path("./checkpoints")
    training_metrics_path: Path = Path("./logs/behavioral_cloning_metrics.csv")
    training_resume_path: Optional[Path] = None
    training_learning_rate: float = 1e-3
    training_weight_decay: float = 1e-4
    training_batch_size: int = 256
    training_epochs: int = 30
    training_num_workers: int = 4
    training_seed: int = 42
    training_scheduler_patience: int = 3
    training_scheduler_factor: float = 0.5
    training_min_learning_rate: float = 1e-6
    training_use_amp: bool = True
    training_deterministic: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Return CORS_ORIGINS as a clean list of strings for FastAPI's middleware."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so the .env file is only parsed once per process."""
    return Settings()


settings = get_settings()
