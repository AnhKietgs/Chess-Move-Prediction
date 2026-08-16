"""
Routes under /api/play/* — the controller layer.

Routes only handle HTTP concerns (lifespan, validation, and HTTP errors).
Neural inference lives in ``src.services.ai_engine``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import APIRouter, FastAPI, HTTPException, Request

from src.models.schemas import MoveRequest
from src.services.ai_engine import FischerAI, NoLegalMovesError, get_fischer_ai


@asynccontextmanager
async def router_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the Fischer policy once before this router begins serving requests.

    Args:
        app: The FastAPI application owning this router.

    Yields:
        Control to FastAPI after the cached inference service is attached to
        application state.
    """
    app.state.fischer_ai = get_fischer_ai()
    yield


router = APIRouter(
    prefix="/api/play",
    tags=["play"],
    lifespan=router_lifespan,
)


@router.post("/fischer")
def play_fischer(payload: MoveRequest, request: Request) -> dict[str, str]:
    """Given the current FEN, return the AI's next move.

    The preloaded Behavioral Cloning policy masks illegal actions before
    selecting one legal UCI move.

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
        move = cast(FischerAI, fischer_ai).predict_best_move(payload.fen)
    except NoLegalMovesError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"move": move}
