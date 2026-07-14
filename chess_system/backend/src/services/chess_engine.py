"""
Chess move generation service.

This module is the seam where the future trained model plugs in.
For now, `select_move` picks a uniformly random legal move — this is the
explicit placeholder the project brief asked for, to be replaced by
`model.predict(fen) -> move` once the Behavioral Cloning network exists
(see /backend/src/training).
"""

import random

import chess


class InvalidFenError(ValueError):
    """Raised when the client sends a FEN string python-chess cannot parse."""


class GameOverError(ValueError):
    """Raised when a move is requested for a position that has no legal moves."""


def _load_board(fen: str) -> chess.Board:
    """Parse a FEN string into a python-chess Board, validating it strictly.

    Args:
        fen: Full FEN string including turn, castling rights, and en-passant.

    Returns:
        A validated `chess.Board` instance.

    Raises:
        InvalidFenError: If the FEN is malformed or structurally illegal.
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise InvalidFenError(f"Malformed FEN string: {exc}") from exc

    if not board.is_valid():
        raise InvalidFenError("FEN parses but represents an illegal chess position.")

    return board


def select_move(fen: str) -> dict:
    """Select a move for the given position and return the resulting state.

    Placeholder strategy: uniformly random choice among all legal moves.
    This will be swapped for `FischerPolicyNet.predict(board_tensor)` once
    the imitation-learning model (Weeks 4-6) is trained and loaded.

    Args:
        fen: Current board state in FEN notation.

    Returns:
        A dict matching the shape of `MoveResponse`.

    Raises:
        InvalidFenError: If the FEN cannot be parsed.
        GameOverError: If the position has no legal moves (checkmate/stalemate).
    """
    board = _load_board(fen)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise GameOverError("No legal moves available; the game has already ended.")

    chosen_move = random.choice(legal_moves)
    move_san = board.san(chosen_move)
    move_uci = chosen_move.uci()

    board.push(chosen_move)

    return {
        "move_uci": move_uci,
        "move_san": move_san,
        "fen_after": board.fen(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_check": board.is_check(),
        "game_over": board.is_game_over(),
    }
