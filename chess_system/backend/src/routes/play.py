"""
Routes under /api/play/* — the controller layer.

Routes only handle HTTP concerns (lifespan, validation, and HTTP errors).
Neural inference lives in ``src.services.ai_engine``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from typing import cast

import chess.engine
from fastapi import APIRouter, FastAPI, HTTPException, Request

from src.models.schemas import MoveRequest
from src.services.ai_engine import (
    FischerAI,
    NoLegalMovesError,
    create_stockfish_engine,
    get_fischer_ai,
)

logger = logging.getLogger(__name__)


def _shutdown_stockfish_engine(engine: chess.engine.SimpleEngine) -> None:
    """Stop Stockfish without failing shutdown when its process already exited.

    Args:
        engine: Stockfish process created for the current FastAPI lifespan.
    """
    try:
        engine.quit()
    except chess.engine.EngineTerminatedError:
        logger.info("Stockfish had already exited before application shutdown.")
    except chess.engine.EngineError:
        logger.warning("Stockfish could not shut down cleanly.", exc_info=True)


@asynccontextmanager
async def router_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the Fischer policy once before this router begins serving requests.

    Args:
        app: The FastAPI application owning this router.

    Yields:
        Control to FastAPI after the cached inference service is attached to
        application state.
    """
    fischer_ai = get_fischer_ai()
    engine = create_stockfish_engine()
    fischer_ai.set_engine(engine)
    app.state.fischer_ai = fischer_ai
    try:
        yield
    finally:
        fischer_ai.clear_engine()
        _shutdown_stockfish_engine(engine)


router = APIRouter(
    prefix="/api/play",
    tags=["play"],
    lifespan=router_lifespan,
)


@router.post("/fischer")
def play_fischer(payload: MoveRequest, request: Request) -> dict[str, str]:
    """Given the current FEN, return the AI's next move.

    The preloaded Behavioral Cloning policy masks illegal actions, then a
    startup-created Stockfish safety-net rejects likely blunders.

    Args:
        payload: Request body containing a full FEN string.
        request: FastAPI request used to retrieve the startup-loaded policy.

    Returns:
        JSON-compatible mapping in the form ``{"move": "e2e4"}``.

    Raises:
        HTTPException: For invalid FEN input, game-over positions, or a policy
            service that has not completed startup.
    """
    fischer_ai = getattr(request.app.state, "fischer_ai", None)
    if fischer_ai is None:
        raise HTTPException(status_code=503, detail="Fischer AI is not initialized.")

    try:
        move = cast(FischerAI, fischer_ai).predict_move(payload.fen)
    except NoLegalMovesError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"move": move.uci()}
