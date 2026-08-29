"""Inference service for selecting legal Fischer policy moves."""

from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
from threading import RLock
from typing import Optional, Union

import chess
import chess.engine
import torch

from src.config.settings import settings
from src.data_processing.encoder import fen_to_tensor, index_to_move, move_to_index
from src.models.chess_model import (
    FischerPolicyNet,
    get_available_device,
    load_model_weights,
    mask_illegal_logits,
)

logger = logging.getLogger(__name__)
_MATE_SCORE_CP = 100_000


class NoLegalMovesError(ValueError):
    """Raised when an inference position has no legal chess moves."""


def create_stockfish_engine(
    stockfish_path: Optional[Union[str, Path]] = None,
) -> chess.engine.SimpleEngine:
    """Start the Stockfish process used by the inference blunder guard.

    Args:
        stockfish_path: Optional engine executable path. The centralized
            ``STOCKFISH_PATH`` setting is used when omitted.

    Returns:
        A running UCI Stockfish engine. The caller owns and must quit it.

    Raises:
        FileNotFoundError: If the configured Stockfish executable is absent.
        chess.engine.EngineError: If the executable cannot start as a UCI engine.
    """
    engine_path = Path(stockfish_path or settings.stockfish_path)
    if not engine_path.is_file():
        raise FileNotFoundError(f"Stockfish executable does not exist: {engine_path}")
    return chess.engine.SimpleEngine.popen_uci(engine_path)


def _validated_board(fen: str) -> chess.Board:
    """Parse a FEN and ensure it represents a valid chess position."""
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Malformed FEN string: {exc}") from exc
    if not board.is_valid():
        raise ValueError("FEN represents an illegal chess position.")
    if not any(board.legal_moves):
        raise NoLegalMovesError("No legal moves are available in this position.")
    return board


def _model_top_moves(
    board: chess.Board,
    model: FischerPolicyNet,
    top_k: int,
) -> list[chess.Move]:
    """Return model-ranked legal moves without creating an autograd graph."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    legal_moves = list(board.legal_moves)
    legal_mask = torch.zeros((1, model.num_actions), dtype=torch.bool, device=device)
    legal_indices = [move_to_index(move) for move in legal_moves]
    legal_mask[0, legal_indices] = True

    board_tensor = fen_to_tensor(board.fen()).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(board_tensor)
    expected_shape = (1, model.num_actions)
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Expected policy logits shape {expected_shape}, got {tuple(logits.shape)}."
        )

    masked_logits = mask_illegal_logits(logits, legal_mask)
    ranked_indices = torch.topk(
        masked_logits,
        k=min(top_k, len(legal_moves)),
        dim=1,
    ).indices[0]
    ranked_moves = [index_to_move(int(index.item()), board) for index in ranked_indices]
    if any(move not in legal_moves for move in ranked_moves):
        raise RuntimeError("Legal move masking produced a non-legal move.")
    return ranked_moves


def _select_safe_move(
    fen: str,
    model: FischerPolicyNet,
    engine: chess.engine.SimpleEngine,
    top_k: int,
    blunder_threshold_cp: int,
    depth: int,
) -> tuple[chess.Move, bool]:
    """Return a guarded move and whether Stockfish fallback was necessary."""
    if blunder_threshold_cp < 0:
        raise ValueError("blunder_threshold_cp must be non-negative.")
    if depth < 1:
        raise ValueError("depth must be at least 1.")

    board = _validated_board(fen)
    mover = board.turn
    limit = chess.engine.Limit(depth=depth)
    best_analysis = engine.analyse(board, limit)
    principal_variation = best_analysis.get("pv", [])
    if not principal_variation:
        raise RuntimeError("Stockfish returned no principal variation.")
    best_move = principal_variation[0]
    score_before = best_analysis["score"].pov(mover).score(mate_score=_MATE_SCORE_CP)
    if score_before is None:
        raise RuntimeError("Stockfish returned an unconvertible position score.")

    for candidate_move in _model_top_moves(board, model, top_k):
        board.push(candidate_move)
        try:
            score_after = engine.analyse(board, limit)["score"].pov(mover).score(
                mate_score=_MATE_SCORE_CP
            )
        finally:
            board.pop()
        if score_after is None:
            raise RuntimeError("Stockfish returned an unconvertible move score.")
        delta_cp = score_after - score_before
        if delta_cp >= -blunder_threshold_cp:
            return candidate_move, False
        logger.debug(
            "Safety-net rejected model move %s (delta_cp=%d, threshold=%d).",
            candidate_move.uci(),
            delta_cp,
            blunder_threshold_cp,
        )
    return best_move, True


def predict_move(
    fen: str,
    model: FischerPolicyNet,
    engine: chess.engine.SimpleEngine,
    top_k: int = 3,
    blunder_threshold_cp: int = 150,
) -> chess.Move:
    """Choose a legal model move guarded by a low-depth Stockfish search.

    The legal-masked policy Top-k moves are evaluated in descending model
    score. A move is rejected when its centipawn change from Stockfish's
    pre-move evaluation is below ``-blunder_threshold_cp``. If every
    candidate is rejected, Stockfish's principal-variation move is returned.

    Args:
        fen: Full FEN string describing the current board position.
        model: Evaluation-mode Fischer policy that emits raw logits.
        engine: Reusable running UCI engine instance.
        top_k: Number of legal policy candidates to inspect.
        blunder_threshold_cp: Maximum allowed centipawn loss.

    Returns:
        A legal chess move selected by the guard.
    """
    selected_move, used_fallback = _select_safe_move(
        fen,
        model,
        engine,
        top_k=top_k,
        blunder_threshold_cp=blunder_threshold_cp,
        depth=settings.inference_stockfish_depth,
    )
    if used_fallback:
        logger.warning(
            "Safety-net fallback: all %d model candidates were blunders; "
            "using Stockfish move %s.",
            top_k,
            selected_move.uci(),
        )
    return selected_move


class FischerAI:
    """Load the trained Fischer policy once and select legal moves from FEN.

    Args:
        model_path: Optional checkpoint path. Defaults to the centralized
            ``MODEL_CHECKPOINT_PATH`` setting.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        engine: Optional[chess.engine.SimpleEngine] = None,
    ) -> None:
        self.device = get_available_device()
        self.model_path = Path(model_path or settings.model_checkpoint_path)
        self.model: FischerPolicyNet = load_model_weights(self.model_path, self.device)
        self.model.eval()
        self.engine = engine
        self._engine_lock = RLock()
        self.safety_net_fallbacks = 0

    def set_engine(self, engine: chess.engine.SimpleEngine) -> None:
        """Attach the startup-created Stockfish engine to this service.

        Args:
            engine: Running Stockfish process shared by API requests.
        """
        with self._engine_lock:
            self.engine = engine

    def clear_engine(self) -> None:
        """Detach a stopped Stockfish engine during application shutdown."""
        with self._engine_lock:
            self.engine = None

    def predict_move(self, fen: str) -> chess.Move:
        """Select a safety-checked legal move for an API request.

        Args:
            fen: Full FEN string describing the current board position.

        Returns:
            A legal move, guarded against Stockfish-detected blunders.

        Raises:
            RuntimeError: If the Stockfish engine was not initialized at startup.
        """
        with self._engine_lock:
            if self.engine is None:
                raise RuntimeError("Stockfish safety-net engine is not initialized.")
            selected_move, used_fallback = _select_safe_move(
                fen,
                self.model,
                self.engine,
                top_k=settings.inference_top_k,
                blunder_threshold_cp=settings.inference_blunder_threshold_cp,
                depth=settings.inference_stockfish_depth,
            )
            if used_fallback:
                self.safety_net_fallbacks += 1
                logger.warning(
                    "Safety-net fallback #%d: all %d model candidates were "
                    "blunders; using Stockfish move %s.",
                    self.safety_net_fallbacks,
                    settings.inference_top_k,
                    selected_move.uci(),
                )
            return selected_move

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
        board = _validated_board(fen)
        return _model_top_moves(board, self.model, top_k=1)[0].uci()


@lru_cache(maxsize=1)
def get_fischer_ai() -> FischerAI:
    """Return the process-wide Fischer policy instance loaded from checkpoint.

    Returns:
        Cached Fischer inference service. The checkpoint is loaded once per
        FastAPI worker process.
    """
    return FischerAI()
