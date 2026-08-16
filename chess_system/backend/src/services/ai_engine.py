"""Inference service for selecting legal Fischer policy moves."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import chess
import torch

from src.config.settings import settings
from src.data_processing.encoder import fen_to_tensor, index_to_move, move_to_index
from src.models.chess_model import FischerPolicyNet, get_available_device, load_model_weights


class NoLegalMovesError(ValueError):
    """Raised when an inference position has no legal chess moves."""


class FischerAI:
    """Load the trained Fischer policy once and select legal moves from FEN.

    Args:
        model_path: Optional checkpoint path. Defaults to the centralized
            ``MODEL_CHECKPOINT_PATH`` setting.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        self.device = get_available_device()
        self.model_path = Path(model_path or settings.model_checkpoint_path)
        self.model: FischerPolicyNet = load_model_weights(self.model_path, self.device)
        self.model.eval()

    def predict_best_move(self, fen: str) -> str:
        """Return the highest-scoring legal UCI move for a chess position.

        Raw policy logits are masked against legal actions before ``argmax``.
        The action encoding and decoding are imported directly from the data
        pipeline, guaranteeing inference uses the same 4,672-action mapping
        as Behavioral Cloning training.

        Args:
            fen: Full FEN string describing the current board position.

        Returns:
            The selected legal move in UCI format, for example ``"e2e4"``.

        Raises:
            ValueError: If ``fen`` is malformed or represents an invalid board.
            NoLegalMovesError: If the position is checkmate or stalemate.
            RuntimeError: If the policy produces an unexpected logits shape.
        """
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise ValueError(f"Malformed FEN string: {exc}") from exc
        if not board.is_valid():
            raise ValueError("FEN represents an illegal chess position.")

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise NoLegalMovesError("No legal moves are available in this position.")

        board_tensor = fen_to_tensor(fen).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(board_tensor)

        expected_shape = (1, self.model.num_actions)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                f"Expected policy logits shape {expected_shape}, got {tuple(logits.shape)}."
            )

        legal_mask = torch.zeros(
            self.model.num_actions,
            dtype=torch.bool,
            device=self.device,
        )
        legal_indices = [move_to_index(move) for move in legal_moves]
        legal_mask[legal_indices] = True
        masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
        action_index = int(torch.argmax(masked_logits, dim=1).item())
        selected_move = index_to_move(action_index, board)

        if selected_move not in legal_moves:
            raise RuntimeError("Legal move masking produced a non-legal move.")
        return selected_move.uci()


@lru_cache(maxsize=1)
def get_fischer_ai() -> FischerAI:
    """Return the process-wide Fischer policy instance loaded from checkpoint.

    Returns:
        Cached Fischer inference service. The checkpoint is loaded once per
        FastAPI worker process.
    """
    return FischerAI()
