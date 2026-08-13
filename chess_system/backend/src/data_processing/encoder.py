"""
Chess state representation for the Behavioral Cloning policy network.

Two independent encodings live here:
    - `fen_to_tensor`: board position -> multi-channel image-like tensor.
    - `move_to_index` / `index_to_move`: chess.Move <-> integer class label,
      for use with `nn.CrossEntropyLoss`.

Both are pure functions with no board-mutation side effects, so they're
safe to call from PyTorch DataLoader worker processes.
"""

from __future__ import annotations

import chess
import torch

# ---------------------------------------------------------------------------
# Board tensor encoding
# ---------------------------------------------------------------------------

_NUM_PIECE_CHANNELS = 12
_NUM_CASTLING_CHANNELS = 4
_NUM_EN_PASSANT_CHANNELS = 1
_NUM_TURN_CHANNELS = 1

NUM_CHANNELS = (
    _NUM_PIECE_CHANNELS + _NUM_CASTLING_CHANNELS + _NUM_EN_PASSANT_CHANNELS + _NUM_TURN_CHANNELS
)  # 18

_PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

CH_WHITE_KINGSIDE = 12
CH_WHITE_QUEENSIDE = 13
CH_BLACK_KINGSIDE = 14
CH_BLACK_QUEENSIDE = 15
CH_EN_PASSANT = 16
CH_SIDE_TO_MOVE = 17


def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Encode a board position (FEN) as a multi-channel tensor.

    Channel layout (18 channels total):
        0-5   : white P, N, B, R, Q, K occupancy planes (1.0 = piece present)
        6-11  : black p, n, b, r, q, k occupancy planes
        12    : white kingside castling right  (constant plane, 1.0 or 0.0)
        13    : white queenside castling right (constant plane)
        14    : black kingside castling right  (constant plane)
        15    : black queenside castling right (constant plane)
        16    : en passant target square (one-hot; all-zero if none)
        17    : side to move (constant plane, 1.0 = White, 0.0 = Black)

    Row/column convention: tensor[channel, rank, file], where rank 0 = "1"
    and file 0 = "a" (so a1 -> tensor[:, 0, 0], h8 -> tensor[:, 7, 7]). No
    perspective flip is applied for Black to move.

    Args:
        fen: Full FEN string (must include turn, castling, en-passant
            fields — not just the piece-placement field).

    Returns:
        Tensor of shape [channels, 8, 8] == [18, 8, 8], dtype=torch.float32.

    Raises:
        ValueError: If `fen` is not a structurally valid FEN (propagated
            from python-chess's `chess.Board` constructor).
    """
    board = chess.Board(fen)
    tensor = torch.zeros((NUM_CHANNELS, 8, 8), dtype=torch.float32)

    for square, piece in board.piece_map().items():
        channel = _PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        tensor[channel, rank, file] = 1.0

    tensor[CH_WHITE_KINGSIDE, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[CH_WHITE_QUEENSIDE, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[CH_BLACK_KINGSIDE, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    tensor[CH_BLACK_QUEENSIDE, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        tensor[CH_EN_PASSANT, ep_rank, ep_file] = 1.0

    tensor[CH_SIDE_TO_MOVE, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    return tensor


# ---------------------------------------------------------------------------
# Move <-> class index encoding (AlphaZero-style 73-plane action space)
# ---------------------------------------------------------------------------

NUM_SQUARES = 64
NUM_MOVE_PLANES = 73
ACTION_SPACE_SIZE = NUM_SQUARES * NUM_MOVE_PLANES  # 4672

# 8 "queen-like" directions as (delta_file, delta_rank): N, NE, E, SE, S, SW, W, NW.
# 7 possible distances (1-7) per direction -> 56 planes. Covers every
# rook/bishop/queen/king move, every non-promoting pawn push/capture, AND
# queen promotions (a queen promotion moves exactly like a 1-square pawn
# push/diagonal-capture, so it naturally lands in one of these 56 planes).
_QUEEN_DIRECTIONS: tuple = (
    (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1),
)
_NUM_QUEEN_PLANES = len(_QUEEN_DIRECTIONS) * 7  # 56

# 8 knight-shaped displacements -> planes 56-63.
_KNIGHT_DELTAS: tuple = (
    (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2),
)
_KNIGHT_PLANE_OFFSET = _NUM_QUEEN_PLANES  # 56

# Underpromotions only (Knight/Bishop/Rook — Queen handled above): 3 forward
# directions x 3 piece choices -> planes 64-72.
_UNDERPROMOTION_DIRECTIONS: tuple = (-1, 0, 1)  # capture-left, straight, capture-right (delta_file)
_UNDERPROMOTION_PIECES: tuple = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
_UNDERPROMOTION_PLANE_OFFSET = _KNIGHT_PLANE_OFFSET + len(_KNIGHT_DELTAS)  # 64

assert _UNDERPROMOTION_PLANE_OFFSET + len(_UNDERPROMOTION_DIRECTIONS) * len(_UNDERPROMOTION_PIECES) == NUM_MOVE_PLANES


def move_to_index(move: chess.Move) -> int:
    """
    Map a chess.Move to an integer class label for CrossEntropyLoss.

    Encoding: `index = from_square * 73 + plane`, where `plane` identifies
    *how* the piece moves from `from_square` (73 possibilities — see the
    module-level `_QUEEN_DIRECTIONS` / `_KNIGHT_DELTAS` / underpromotion
    tables above). This operates directly on `move.from_square`/`to_square`
    with no board-perspective flip for Black, consistent with
    `fen_to_tensor`'s absolute (non-mirrored) board representation.

    Args:
        move: A `chess.Move`. Must correspond to a geometrically valid
            piece-movement pattern (straight line, diagonal, or knight
            shape) — i.e. anything a real chess piece could move.

    Returns:
        Integer class label, 0 <= index < ACTION_SPACE_SIZE (4672).

    Raises:
        ValueError: If `move` doesn't correspond to any representable
            movement pattern — i.e. it falls outside the action space.
    """
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)

    delta_file = to_file - from_file
    delta_rank = to_rank - from_rank

    if delta_file == 0 and delta_rank == 0:
        raise ValueError(f"Move {move.uci()} has zero displacement; not in action space.")

    if move.promotion is not None and move.promotion != chess.QUEEN:
        if delta_file not in _UNDERPROMOTION_DIRECTIONS or abs(delta_rank) != 1:
            raise ValueError(f"Move {move.uci()} is not a valid pawn underpromotion; not in action space.")
        if move.promotion not in _UNDERPROMOTION_PIECES:
            raise ValueError(f"Unsupported underpromotion piece in move {move.uci()}.")
        direction_idx = _UNDERPROMOTION_DIRECTIONS.index(delta_file)
        piece_idx = _UNDERPROMOTION_PIECES.index(move.promotion)
        plane = _UNDERPROMOTION_PLANE_OFFSET + direction_idx * len(_UNDERPROMOTION_PIECES) + piece_idx
        return move.from_square * NUM_MOVE_PLANES + plane

    if (delta_file, delta_rank) in _KNIGHT_DELTAS:
        plane = _KNIGHT_PLANE_OFFSET + _KNIGHT_DELTAS.index((delta_file, delta_rank))
        return move.from_square * NUM_MOVE_PLANES + plane

    distance = max(abs(delta_file), abs(delta_rank))
    if distance > 7 or delta_file % distance != 0 or delta_rank % distance != 0:
        raise ValueError(
            f"Move {move.uci()} is not a straight-line, diagonal, or knight-shaped move; "
            "not in action space."
        )
    direction = (delta_file // distance, delta_rank // distance)
    if direction not in _QUEEN_DIRECTIONS:
        raise ValueError(f"Move {move.uci()} has an unrecognized direction {direction}.")

    plane = _QUEEN_DIRECTIONS.index(direction) * 7 + (distance - 1)
    return move.from_square * NUM_MOVE_PLANES + plane


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Inverse of `move_to_index`.

    `board` is only needed to disambiguate an ordinary pawn push from a
    queen promotion, since both share the same "queen-direction" planes —
    we check whether the piece on `from_square` is a pawn landing on the
    back rank to decide.

    Args:
        index: Class label in [0, ACTION_SPACE_SIZE).
        board: Board the move would be played on.

    Returns:
        The decoded `chess.Move`. NOT guaranteed to be legal on `board` —
        callers must still check `move in board.legal_moves` before using it
        (e.g. model logits can point at squares with no piece, pinned
        pieces, etc.).

    Raises:
        ValueError: If `index` is outside [0, ACTION_SPACE_SIZE), or decodes
            to an off-board target square.
    """
    if not (0 <= index < ACTION_SPACE_SIZE):
        raise ValueError(f"Index {index} is outside the action space [0, {ACTION_SPACE_SIZE}).")

    from_square, plane = divmod(index, NUM_MOVE_PLANES)
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)

    if plane < _KNIGHT_PLANE_OFFSET:
        direction_idx, distance_idx = divmod(plane, 7)
        delta_file, delta_rank = _QUEEN_DIRECTIONS[direction_idx]
        distance = distance_idx + 1
        to_file = from_file + delta_file * distance
        to_rank = from_rank + delta_rank * distance
        promotion = None
        piece = board.piece_at(from_square)
        if piece is not None and piece.piece_type == chess.PAWN and to_rank in (0, 7):
            promotion = chess.QUEEN
    elif plane < _UNDERPROMOTION_PLANE_OFFSET:
        delta_file, delta_rank = _KNIGHT_DELTAS[plane - _KNIGHT_PLANE_OFFSET]
        to_file = from_file + delta_file
        to_rank = from_rank + delta_rank
        promotion = None
    else:
        under_idx = plane - _UNDERPROMOTION_PLANE_OFFSET
        direction_idx, piece_idx = divmod(under_idx, len(_UNDERPROMOTION_PIECES))
        delta_file = _UNDERPROMOTION_DIRECTIONS[direction_idx]
        piece = board.piece_at(from_square)
        delta_rank = 1 if (piece is not None and piece.color == chess.WHITE) else -1
        to_file = from_file + delta_file
        to_rank = from_rank + delta_rank
        promotion = _UNDERPROMOTION_PIECES[piece_idx]

    if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
        raise ValueError(f"Index {index} decodes to an off-board square; not a valid move.")

    to_square = chess.square(to_file, to_rank)
    return chess.Move(from_square, to_square, promotion=promotion)
