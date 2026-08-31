"""Analytics endpoints used by the Fischer style-verification dashboard."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from io import StringIO
from typing import Any, Sequence

import chess
import chess.pgn
import torch
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from src.config.settings import settings
from src.data_processing.dataset import (
    _Record,
    _load_records,
    enforce_fen_disjoint_splits,
    split_indices_by_game,
)
from src.data_processing.encoder import fen_to_tensor, index_to_move, move_to_index
from src.models.chess_model import FischerPolicyNet, mask_illegal_logits
from src.services.ai_engine import FischerAI, get_fischer_ai

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@lru_cache(maxsize=1)
def _heldout_records(strict_fen_disjoint: bool = True) -> tuple[_Record, ...]:
    """Return deterministic test records from the Behavioral Cloning split.

    Strict FEN filtering is appropriate for model-generalization metrics and
    qualitative move examples. Standard first-opening positions are shared
    by many games, however, so opening-frequency charts intentionally use a
    game-held-out split without this additional filtering.

    Returns:
        Immutable held-out cache records in deterministic split order.

    Raises:
        FileNotFoundError: If the configured training cache is absent.
    """
    cache_path = settings.training_data_path
    records = _load_records(cache_path)
    splits = split_indices_by_game(records, seed=settings.training_seed)
    if strict_fen_disjoint:
        splits, _ = enforce_fen_disjoint_splits(records, splits)
    return tuple(records[index] for index in splits["test"])


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
    records: Sequence[_Record],
    model: FischerPolicyNet,
    limit: int,
    fischer_color: chess.Color,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Build held-out opening distributions for Fischer playing one color."""
    actual_counts: Counter[str] = Counter()
    ai_probability_mass: Counter[str] = Counter()
    samples = 0

    for record in records:
        try:
            board = chess.Board(record.fen)
            recorded_move = chess.Move.from_uci(record.move_uci)
        except ValueError:
            continue

        # Compare opening choices at Fischer's first turn only. For Black,
        # this is the reply after White's opening move, so labels use the
        # conventional ``1...`` notation.
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
    fischer_color = chess.WHITE if color == "white" else chess.BLACK
    try:
        # Identical opening FENs naturally repeat across distinct games, so
        # strict FEN filtering would remove all first-move examples. A
        # game-held-out set still prevents whole games from crossing splits.
        records = _heldout_records(strict_fen_disjoint=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    actual, ai, samples = _opening_distribution(records, _get_policy(request), limit, fischer_color)
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
        "source": "game-held-out opening positions; repeated opening FENs are unavoidable",
    }


def _heldout_position_summary(
    record: _Record,
    model: FischerPolicyNet,
) -> dict[str, Any] | None:
    """Summarize one recorded Fischer move and the model's legal Top-3.

    Args:
        record: One held-out Fischer state/action record.
        model: Evaluation-mode policy shared by the FastAPI application.

    Returns:
        A JSON-compatible comparison, or ``None`` for an invalid record.
    """
    try:
        board = chess.Board(record.fen)
        actual_move = chess.Move.from_uci(record.move_uci)
    except ValueError:
        return None
    if actual_move not in board.legal_moves:
        return None
    predicted_moves = _ranked_legal_moves(board, model, top_k=3)
    if not predicted_moves:
        return None
    return {
        "fen": record.fen,
        "fullmove_number": board.fullmove_number,
        "fischer_color": "white" if board.turn == chess.WHITE else "black",
        "actual_move": {"uci": actual_move.uci(), "san": board.san(actual_move)},
        "ai_top_moves": [
            {"uci": move.uci(), "san": board.san(move)} for move in predicted_moves
        ],
        "top1_match": actual_move == predicted_moves[0],
        "top3_match": actual_move in predicted_moves,
    }


@router.get("/heldout_examples")
def heldout_examples(
    request: Request,
    games: int = Query(default=3, ge=1, le=5),
    positions_per_game: int = Query(default=3, ge=1, le=5),
) -> dict[str, Any]:
    """Return deterministic, uncurated Fischer/AI examples from held-out games.

    The first valid positions from the first held-out game IDs are returned
    deterministically. They are not selected because the model matched them,
    which makes the examples suitable as qualitative supporting evidence.

    Args:
        request: Request used to reuse the startup-loaded Fischer policy.
        games: Number of held-out games to include.
        positions_per_game: Fischer decisions included per returned game.

    Returns:
        Game groups containing Fischer's actual move and AI legal Top-3 moves.
    """
    try:
        records = _heldout_records(strict_fen_disjoint=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    grouped_records: dict[int, list[_Record]] = {}
    for record in records:
        grouped_records.setdefault(record.game_id, []).append(record)

    selected_games: list[dict[str, Any]] = []
    model = _get_policy(request)
    for game_id in sorted(grouped_records):
        positions: list[dict[str, Any]] = []
        for record in grouped_records[game_id]:
            summary = _heldout_position_summary(record, model)
            if summary is not None:
                positions.append(summary)
            if len(positions) == positions_per_game:
                break
        if positions:
            selected_games.append({"game_id": game_id, "positions": positions})
        if len(selected_games) == games:
            break

    if not selected_games:
        raise HTTPException(status_code=422, detail="No valid held-out examples are available.")
    return {
        "games": selected_games,
        "source": "strict held-out split; examples selected by game order, not match rate",
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
