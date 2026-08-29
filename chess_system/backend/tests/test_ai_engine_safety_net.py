"""Unit tests for the inference-time Stockfish blunder guard."""

from __future__ import annotations

import chess
import chess.engine
import torch
from torch import Tensor, nn

from src.data_processing.encoder import ACTION_SPACE_SIZE, move_to_index
from src.services.ai_engine import predict_move


class _FixedPolicy(nn.Module):
    """Minimal policy returning fixed logits for selected UCI moves."""

    def __init__(self, ranked_moves: list[str]) -> None:
        super().__init__()
        self.num_actions = ACTION_SPACE_SIZE
        self.register_buffer("logits", torch.full((1, ACTION_SPACE_SIZE), -10.0))
        for score, move_uci in enumerate(reversed(ranked_moves), start=1):
            self.logits[0, move_to_index(chess.Move.from_uci(move_uci))] = float(score)

    def forward(self, board_tensor: Tensor) -> Tensor:
        """Return logits on the same device as the provided board tensor."""
        return self.logits.to(board_tensor.device).expand(board_tensor.size(0), -1)


class _FakeEngine:
    """Deterministic Stockfish substitute returning scores by played move."""

    def __init__(self, after_move_scores: dict[str, int], best_move: str) -> None:
        self.after_move_scores = after_move_scores
        self.best_move = chess.Move.from_uci(best_move)

    def analyse(self, board: chess.Board, limit: chess.engine.Limit) -> dict[str, object]:
        """Return a zero pre-move score or configured score after a move."""
        del limit
        score = 0
        if board.move_stack:
            score = self.after_move_scores.get(board.peek().uci(), 0)
        return {
            "score": chess.engine.PovScore(chess.engine.Cp(score), chess.WHITE),
            "pv": [self.best_move],
        }


def test_safety_net_skips_blunder_and_uses_next_model_move() -> None:
    """A top-ranked policy blunder is rejected in favor of the next candidate."""
    model = _FixedPolicy(["e2e4", "d2d4", "g1f3"])
    engine = _FakeEngine({"e2e4": -300, "d2d4": -80}, best_move="g1f3")

    selected_move = predict_move(
        chess.STARTING_FEN,
        model,
        engine,  # type: ignore[arg-type]
        top_k=3,
        blunder_threshold_cp=150,
    )

    assert selected_move == chess.Move.from_uci("d2d4")


def test_safety_net_falls_back_to_stockfish_after_all_blunders() -> None:
    """All rejected policy candidates result in Stockfish's best move."""
    model = _FixedPolicy(["e2e4", "d2d4", "g1f3"])
    engine = _FakeEngine(
        {"e2e4": -300, "d2d4": -250, "g1f3": -200},
        best_move="b1c3",
    )

    selected_move = predict_move(
        chess.STARTING_FEN,
        model,
        engine,  # type: ignore[arg-type]
        top_k=3,
        blunder_threshold_cp=150,
    )

    assert selected_move == chess.Move.from_uci("b1c3")
