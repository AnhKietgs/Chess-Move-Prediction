"""Safety tests for horizontal board-mirroring augmentation."""

from __future__ import annotations

import chess
import torch

from src.data_processing.dataset import ChessDataset, _Record
from src.data_processing.encoder import (
    ACTION_SPACE_SIZE,
    board_to_tensor,
    mirror_fen,
    mirror_move,
    move_to_index,
)


def test_mirror_augmentation_preserves_legal_moves_without_castling(
    monkeypatch,
) -> None:
    """Mirror a no-castling en-passant position into an equivalent legal state."""
    fen = "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1"
    source_board = chess.Board(fen)
    source_move = chess.Move.from_uci("e5d6")
    mirrored_board = chess.Board(mirror_fen(fen))
    mirrored_move = mirror_move(source_move)

    monkeypatch.setattr("src.data_processing.dataset.random.random", lambda: 0.0)
    dataset = ChessDataset([_Record(fen, source_move.uci(), 1)], augment=True)
    board_tensor, label, legal_mask = dataset[0]

    expected_moves = {mirror_move(move) for move in source_board.legal_moves}
    assert expected_moves == set(mirrored_board.legal_moves)
    assert mirrored_board.is_valid()
    assert torch.equal(board_tensor, board_to_tensor(mirrored_board))
    assert label.item() == move_to_index(mirrored_move)
    assert legal_mask[label.item()]
    assert 0 <= label.item() < ACTION_SPACE_SIZE


def test_mirror_augmentation_skips_positions_with_castling_rights(monkeypatch) -> None:
    """Keep castling positions unchanged because their horizontal mirror is unsafe."""
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    source_board = chess.Board(fen)
    source_move = chess.Move.from_uci("e1g1")

    monkeypatch.setattr("src.data_processing.dataset.random.random", lambda: 0.0)
    dataset = ChessDataset([_Record(fen, source_move.uci(), 1)], augment=True)
    board_tensor, label, legal_mask = dataset[0]

    assert torch.equal(board_tensor, board_to_tensor(source_board))
    assert label.item() == move_to_index(source_move)
    assert legal_mask[label.item()]
    assert not legal_mask[move_to_index(mirror_move(source_move))]
