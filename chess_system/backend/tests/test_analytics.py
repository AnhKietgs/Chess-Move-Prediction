"""Tests for legal-masked analytics helpers."""

import chess
import chess.pgn
import torch
from torch import Tensor, nn

from src.data_processing.encoder import ACTION_SPACE_SIZE, move_to_index
from src.routes.analytics import _fischer_color, _ranked_legal_moves


class _FixedPolicy(nn.Module):
    """Small policy stub with a high score assigned to a chosen action."""

    def __init__(self, scores: dict[str, float]) -> None:
        super().__init__()
        self.num_actions = ACTION_SPACE_SIZE
        self.anchor = nn.Parameter(torch.zeros(()))
        self.logits = torch.full((1, ACTION_SPACE_SIZE), -20.0)
        for move_uci, score in scores.items():
            self.logits[0, move_to_index(chess.Move.from_uci(move_uci))] = score

    def forward(self, board_tensor: Tensor) -> Tensor:
        """Return fixed policy logits on the board tensor's device."""
        return self.logits.to(board_tensor.device).expand(board_tensor.size(0), -1)


def test_ranked_legal_moves_ignores_high_score_illegal_action() -> None:
    """Analytics must use the same legal-action mask as game inference."""
    board = chess.Board()
    model = _FixedPolicy({"a1a2": 100.0, "e2e4": 5.0, "d2d4": 4.0})

    ranked_moves = _ranked_legal_moves(board, model, top_k=2)  # type: ignore[arg-type]

    assert [move.uci() for move in ranked_moves] == ["e2e4", "d2d4"]


def test_fischer_color_is_found_case_insensitively() -> None:
    """PGN evaluation identifies Fischer's color independently of header case."""
    game = chess.pgn.Game()
    game.headers["White"] = "Robert J. Fischer"
    game.headers["Black"] = "Opponent"

    assert _fischer_color(game, "fischer") == chess.WHITE
