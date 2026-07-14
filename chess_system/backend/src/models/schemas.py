"""
Pydantic schemas shared across routes.

Keeping these separate from `services/` keeps the "shape of data crossing
the wire" decoupled from "logic that operates on the data" — schemas here
should stay free of chess logic.
"""

from typing import Optional

from pydantic import BaseModel, Field


class MoveRequest(BaseModel):
    """Payload sent by the frontend when it wants the AI to produce a move.

    `fen` is the full FEN string of the current board, including active
    color, castling rights, en-passant target, and move clocks — all of
    which are required to generate a rules-correct move.
    """

    fen: str = Field(
        ...,
        description="Current board state in Forsyth-Edwards Notation.",
        examples=["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
    )


class MoveResponse(BaseModel):
    """Payload returned by the AI move endpoint."""

    move_uci: str = Field(..., description="Chosen move in UCI format, e.g. 'e2e4'.")
    move_san: str = Field(..., description="Same move in Standard Algebraic Notation, e.g. 'e4'.")
    fen_after: str = Field(..., description="Resulting FEN after the move is applied.")
    is_checkmate: bool = False
    is_stalemate: bool = False
    is_check: bool = False
    game_over: bool = False


class ErrorResponse(BaseModel):
    """Standard error shape returned to the frontend on 4xx/5xx responses."""

    detail: str
    code: Optional[str] = None
