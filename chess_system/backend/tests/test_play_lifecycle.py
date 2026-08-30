"""Tests for resilient FastAPI Stockfish lifecycle cleanup."""

import chess.engine

from src.routes.play import _shutdown_stockfish_engine


class _AlreadyStoppedEngine:
    """Engine stub that simulates Stockfish exiting before shutdown."""

    def quit(self) -> None:
        """Raise the same exception as python-chess for a dead event loop."""
        raise chess.engine.EngineTerminatedError("engine event loop dead")


def test_shutdown_ignores_engine_already_terminated() -> None:
    """A dead Stockfish process must not make the FastAPI lifespan fail."""
    _shutdown_stockfish_engine(_AlreadyStoppedEngine())  # type: ignore[arg-type]
