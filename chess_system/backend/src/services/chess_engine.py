"""
Chess move generation service.

Uses the trained Fischer Behavioral Cloning policy through ``ai_engine``.
The service preserves the API response contract while replacing the former
random-move placeholder with legal-move-masked neural inference.
"""

import chess

from src.services.ai_engine import get_fischer_ai


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

    The trained Fischer policy scores all actions, masks every illegal action,
    and returns the highest-scoring legal move.

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

    move_uci = get_fischer_ai().predict_best_move(fen)
    chosen_move = chess.Move.from_uci(move_uci)
    if chosen_move not in legal_moves:
        raise RuntimeError("Fischer policy returned a non-legal move.")
    move_san = board.san(chosen_move)

    board.push(chosen_move)

    return {
        "move_uci": chosen_move.uci(),
        "move_san": move_san,
        "fen_after": board.fen(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
        "is_check": board.is_check(),
        "game_over": board.is_game_over(),
    }
