"""
Routes under /api/play/* — the controller layer.

Routes only handle HTTP concerns (status codes, request/response schemas).
Actual chess logic lives in `src/services/chess_engine.py`.
"""

from fastapi import APIRouter, HTTPException

from src.models.schemas import MoveRequest, MoveResponse
from src.services import chess_engine

router = APIRouter(prefix="/api/play", tags=["play"])


@router.post("/fischer", response_model=MoveResponse)
def play_fischer(payload: MoveRequest) -> MoveResponse:
    """Given the current FEN, return the AI's next move.

    Currently backed by a random-legal-move placeholder
    (see `chess_engine.select_move`); will be swapped for the trained
    Behavioral Cloning policy without changing this route's contract.
    """
    try:
        result = chess_engine.select_move(payload.fen)
    except chess_engine.InvalidFenError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except chess_engine.GameOverError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return MoveResponse(**result)
