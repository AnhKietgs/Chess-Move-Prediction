"""Analytics endpoints used by the Fischer style-verification dashboard."""

from __future__ import annotations

from collections import Counter
from io import StringIO
import json
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import torch
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from src.config.settings import settings
from src.data_processing.encoder import fen_to_tensor, index_to_move, move_to_index
from src.models.chess_model import FischerPolicyNet, mask_illegal_logits
from src.services.ai_engine import FischerAI, get_fischer_ai

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _get_policy(request: Request) -> FischerPolicyNet:
    """Return the already-loaded policy model for an analytics request."""
    fischer_ai = getattr(request.app.state, "fischer_ai", None)
    if fischer_ai is None:
        fischer_ai = get_fischer_ai()
    if not isinstance(fischer_ai, FischerAI):
        raise HTTPException(status_code=503, detail="Fischer AI is not initialized.")
    return fischer_ai.model


def _ranked_legal_moves(
    board: chess.Board,
    model: FischerPolicyNet,
    top_k: int,
) -> list[chess.Move]:
    """Rank a board's legal moves by raw policy logits.

    Args:
        board: Valid board whose legal moves are ranked.
        model: Evaluation-mode Fischer policy.
        top_k: Number of legal actions to return.

    Returns:
        Legal moves ordered from highest to lowest policy score.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []

    device = next(model.parameters()).device
    legal_mask = torch.zeros((1, model.num_actions), dtype=torch.bool, device=device)
    legal_indices = [move_to_index(move) for move in legal_moves]
    legal_mask[0, legal_indices] = True
    board_tensor = fen_to_tensor(board.fen()).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(board_tensor)
    masked_logits = mask_illegal_logits(logits, legal_mask)
    action_indices = torch.topk(
        masked_logits,
        k=min(top_k, len(legal_moves)),
        dim=1,
    ).indices[0]
    return [index_to_move(int(index.item()), board) for index in action_indices]


def _opening_distribution(
    cache_path: Path,
    model: FischerPolicyNet,
    limit: int,
    fischer_color: chess.Color,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Build opening distributions for Fischer playing one specified color."""
    actual_counts: Counter[str] = Counter()
    ai_probability_mass: Counter[str] = Counter()
    samples = 0

    with cache_path.open("r", encoding="utf-8") as cache_file:
        for line_number, line in enumerate(cache_file, start=1):
            try:
                payload = json.loads(line)
                board = chess.Board(payload["fen"])
                recorded_move = chess.Move.from_uci(payload["move_uci"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

            # Compare opening choices at Fischer's first turn only. For
            # Black, this is the reply after White's opening move, so labels
            # use the conventional ``1...`` notation.
            if board.fullmove_number != 1 or board.turn != fischer_color:
                continue
            if recorded_move not in board.legal_moves:
                continue

            move_prefix = "1." if fischer_color == chess.WHITE else "1..."
            actual_counts[f"{move_prefix} {board.san(recorded_move)}"] += 1
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                continue

            device = next(model.parameters()).device
            legal_mask = torch.zeros((1, model.num_actions), dtype=torch.bool, device=device)
            legal_indices = [move_to_index(move) for move in legal_moves]
            legal_mask[0, legal_indices] = True
            board_tensor = fen_to_tensor(board.fen()).unsqueeze(0).to(device)
            with torch.no_grad():
                masked_logits = mask_illegal_logits(model(board_tensor), legal_mask)
                probabilities = torch.softmax(masked_logits, dim=1)[0]
            for move, index in zip(legal_moves, legal_indices):
                ai_probability_mass[
                    f"{move_prefix} {board.san(move)}"
                ] += float(probabilities[index].item())
            samples += 1

    def to_percentages(counts: Counter[str]) -> list[dict[str, Any]]:
        total = sum(counts.values())
        if total == 0:
            return []
        return [
            {
                "move": move,
                "value": round(value / total * 100, 2),
                "count": round(value, 4),
            }
            for move, value in counts.most_common(limit)
        ]

    return to_percentages(actual_counts), to_percentages(ai_probability_mass), samples


def _fischer_color(game: chess.pgn.Game, player_name: str) -> chess.Color | None:
    """Return Fischer's color in one PGN game, if the player occurs in headers."""
    player = player_name.casefold()
    if player in game.headers.get("White", "").casefold():
        return chess.WHITE
    if player in game.headers.get("Black", "").casefold():
        return chess.BLACK
    return None


@router.get("/opening_stats")
def opening_stats(
    request: Request,
    limit: int = Query(default=6, ge=1, le=20),
    color: str = Query(default="white", pattern="^(white|black)$"),
) -> dict[str, Any]:
    """Return Fischer and AI opening distributions for the dashboard.

    The actual distribution comes from cached Fischer training examples. The
    AI distribution is its legal-masked probability mass at the same opening
    positions, rather than a one-off argmax that would always show one move.

    Args:
        request: Request used to reuse the startup-loaded Fischer policy.
        limit: Maximum opening categories returned per distribution.
        color: Fischer's color: ``white`` for first moves or ``black`` for
            first-move replies.

    Returns:
        Opening percentages for Fischer's recorded moves and the AI policy.
    """
    cache_path = settings.training_data_path
    if not cache_path.is_file():
        raise HTTPException(status_code=404, detail=f"Training cache not found: {cache_path}")

    fischer_color = chess.WHITE if color == "white" else chess.BLACK
    actual, ai, samples = _opening_distribution(
        cache_path,
        _get_policy(request),
        limit,
        fischer_color,
    )
    if samples == 0:
        raise HTTPException(
            status_code=422,
            detail=f"No Fischer {color} opening moves found in cache.",
        )
    return {
        "actual": actual,
        "ai": ai,
        "sample_size": samples,
        "fischer_color": color,
        "analysis_type": "opening" if fischer_color == chess.WHITE else "defense",
        "source": "cached Fischer opening positions",
    }


@router.post("/evaluate_pgn")
async def evaluate_pgn(
    request: Request,
    pgn_file: UploadFile = File(...),
    player_name: str = Query(default="Fischer", min_length=1, max_length=100),
) -> dict[str, Any]:
    """Evaluate Top-1 and Top-3 policy agreement against an uploaded PGN.

    Only moves by ``player_name`` are scored, matching the Behavioral
    Cloning training pipeline. The endpoint does not run Stockfish, so it
    evaluates a PGN promptly while retaining the exact legal-action mask
    used at inference.

    Args:
        request: Request used to reuse the startup-loaded Fischer policy.
        pgn_file: Uploaded PGN file containing held-out Fischer games.
        player_name: Case-insensitive player name matched in PGN headers.

    Returns:
        Top-1 and Top-3 agreement rates plus evaluated game/position counts.
    """
    if not pgn_file.filename or not pgn_file.filename.lower().endswith(".pgn"):
        raise HTTPException(status_code=422, detail="Upload a file with a .pgn extension.")
    try:
        pgn_text = (await pgn_file.read()).decode("utf-8", errors="replace")
    finally:
        await pgn_file.close()

    model = _get_policy(request)
    pgn_stream = StringIO(pgn_text)
    top1_matches = 0
    top3_matches = 0
    positions_evaluated = 0
    games_evaluated = 0

    while True:
        try:
            game = chess.pgn.read_game(pgn_stream)
        except Exception as exc:
            raise HTTPException(status_code=422, detail="PGN contains an unreadable game.") from exc
        if game is None:
            break
        target_color = _fischer_color(game, player_name)
        if target_color is None:
            continue

        board = game.board()
        game_positions = 0
        for move in game.mainline_moves():
            if move not in board.legal_moves:
                break
            if board.turn == target_color:
                ranked_moves = _ranked_legal_moves(board, model, top_k=3)
                if ranked_moves:
                    top1_matches += int(move == ranked_moves[0])
                    top3_matches += int(move in ranked_moves)
                    positions_evaluated += 1
                    game_positions += 1
            board.push(move)
        games_evaluated += int(game_positions > 0)

    if positions_evaluated == 0:
        raise HTTPException(
            status_code=422,
            detail=f"No legal moves by '{player_name}' were found in the uploaded PGN.",
        )
    return {
        "top1_match_rate": round(top1_matches / positions_evaluated * 100, 2),
        "top3_match_rate": round(top3_matches / positions_evaluated * 100, 2),
        "positions_evaluated": positions_evaluated,
        "games_evaluated": games_evaluated,
        "player_name": player_name,
    }
