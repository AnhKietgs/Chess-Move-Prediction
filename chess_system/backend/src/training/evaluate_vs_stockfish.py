"""Run a reproducible Fischer-policy arena against limited-strength Stockfish.

Run from the ``backend`` directory with ``python -m
src.training.evaluate_vs_stockfish``. The script loads the Behavioral
Cloning checkpoint once, alternates colors, uses varied opening positions,
and writes game PGNs plus a CSV summary for later analysis.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import chess
import chess.engine
import chess.pgn
from tqdm.auto import tqdm

from src.config.settings import Settings, settings
from src.services.ai_engine import FischerAI


_OPENINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("King's Pawn Game", ("e2e4", "e7e5", "g1f3", "b8c6")),
    ("Sicilian Defence", ("e2e4", "c7c5", "g1f3", "d7d6")),
    ("Caro-Kann Defence", ("e2e4", "c7c6", "d2d4", "d7d5")),
    ("French Defence", ("e2e4", "e7e6", "d2d4", "d7d5")),
    ("Queen's Gambit", ("d2d4", "d7d5", "c2c4", "e7e6")),
    ("King's Indian Defence", ("d2d4", "g8f6", "c2c4", "g7g6")),
    ("English Opening", ("c2c4", "e7e5", "b1c3", "g8f6")),
    ("Reti Opening", ("g1f3", "d7d5", "g2g3", "g8f6")),
)


@dataclass(frozen=True)
class ArenaGameResult:
    """Result and metadata for one FischerAI-versus-Stockfish game."""

    game_number: int
    opening: str
    model_color: str
    result: str
    model_score: float
    plies: int


@dataclass(frozen=True)
class ArenaSummary:
    """Aggregate results and artifact paths for an arena run."""

    games: int
    wins: int
    draws: int
    losses: int
    score: float
    effective_stockfish_elo: int
    pgn_path: Path
    csv_path: Path


def _opening_board(uci_moves: Sequence[str]) -> tuple[chess.Board, list[chess.Move]]:
    """Build a legal board position by replaying an opening UCI sequence."""
    board = chess.Board()
    moves: list[chess.Move] = []
    for uci_move in uci_moves:
        move = chess.Move.from_uci(uci_move)
        if move not in board.legal_moves:
            raise ValueError(f"Configured opening contains an illegal move: {uci_move}")
        board.push(move)
        moves.append(move)
    return board, moves


def _configure_limited_stockfish(
    engine: chess.engine.SimpleEngine,
    requested_elo: int,
) -> int:
    """Enable Stockfish strength limiting and return the supported Elo used."""
    required_options = {"UCI_LimitStrength", "UCI_Elo"}
    missing_options = required_options.difference(engine.options)
    if missing_options:
        raise RuntimeError(
            "The selected engine does not support Stockfish Elo limiting; "
            f"missing options: {sorted(missing_options)}"
        )

    elo_option = engine.options["UCI_Elo"]
    if elo_option.min is None or elo_option.max is None:
        raise RuntimeError("Stockfish UCI_Elo option does not expose a supported range.")
    effective_elo = max(elo_option.min, min(requested_elo, elo_option.max))
    engine.configure(
        {
            "UCI_LimitStrength": True,
            "UCI_Elo": effective_elo,
        }
    )
    return effective_elo


def _result_for_model(
    board: chess.Board,
    model_color: chess.Color,
) -> tuple[str, float]:
    """Return the PGN result and FischerAI score from the final board state."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "1/2-1/2", 0.5
    if outcome.winner == model_color:
        return ("1-0", 1.0) if model_color == chess.WHITE else ("0-1", 1.0)
    return ("0-1", 0.0) if model_color == chess.WHITE else ("1-0", 0.0)


def _play_game(
    game_number: int,
    fischer_ai: FischerAI,
    engine: chess.engine.SimpleEngine,
    model_color: chess.Color,
    opening_name: str,
    opening_moves: Sequence[str],
    stockfish_time_seconds: float,
    max_plies: int,
    stockfish_elo: int,
) -> tuple[ArenaGameResult, chess.pgn.Game]:
    """Play and record one full arena game from a fixed opening position."""
    board, opening_history = _opening_board(opening_moves)
    game = chess.pgn.Game()
    game.headers["Event"] = "FischerAI vs limited Stockfish"
    game.headers["Round"] = str(game_number)
    game.headers["White"] = "FischerAI" if model_color == chess.WHITE else "Stockfish"
    game.headers["Black"] = "FischerAI" if model_color == chess.BLACK else "Stockfish"
    game.headers["Opening"] = opening_name
    game.headers["StockfishElo"] = str(stockfish_elo)

    node = game
    for move in opening_history:
        node = node.add_variation(move)

    additional_plies = 0
    while not board.is_game_over(claim_draw=True) and additional_plies < max_plies:
        if board.turn == model_color:
            move = chess.Move.from_uci(fischer_ai.predict_best_move(board.fen()))
        else:
            engine_result = engine.play(
                board,
                chess.engine.Limit(time=stockfish_time_seconds),
            )
            move = engine_result.move

        if move not in board.legal_moves:
            raise RuntimeError(f"Arena participant returned illegal move: {move.uci()}")
        board.push(move)
        node = node.add_variation(move)
        additional_plies += 1

    result, model_score = _result_for_model(board, model_color)
    game.headers["Result"] = result
    return (
        ArenaGameResult(
            game_number=game_number,
            opening=opening_name,
            model_color="white" if model_color == chess.WHITE else "black",
            result=result,
            model_score=model_score,
            plies=len(opening_history) + additional_plies,
        ),
        game,
    )


def _write_artifacts(
    output_dir: Path,
    run_label: str,
    results: Sequence[ArenaGameResult],
    games: Sequence[chess.pgn.Game],
) -> tuple[Path, Path]:
    """Write the completed arena's PGN game records and CSV result rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pgn_path = output_dir / f"stockfish_arena_{run_label}.pgn"
    csv_path = output_dir / f"stockfish_arena_{run_label}.csv"

    with pgn_path.open("w", encoding="utf-8") as pgn_file:
        for game in games:
            print(game, file=pgn_file, end="\n\n")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "game_number",
                "opening",
                "model_color",
                "result",
                "model_score",
                "plies",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "game_number": result.game_number,
                    "opening": result.opening,
                    "model_color": result.model_color,
                    "result": result.result,
                    "model_score": result.model_score,
                    "plies": result.plies,
                }
            )
    return pgn_path, csv_path


def run_stockfish_arena(
    config: Settings,
    games: int,
    requested_stockfish_elo: int,
    stockfish_time_seconds: float,
    max_plies: int,
    seed: int,
    stockfish_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> ArenaSummary:
    """Measure FischerAI's practical playing strength against Stockfish.

    Args:
        config: Centralized application settings.
        games: Number of games to play; colors alternate from one game to next.
        requested_stockfish_elo: Requested UCI Elo limit for Stockfish.
        stockfish_time_seconds: Stockfish think time per move in seconds.
        max_plies: Maximum post-opening plies before forcing a draw.
        seed: Seed used to randomize opening order reproducibly.
        stockfish_path: Optional UCI engine executable override.
        checkpoint_path: Optional Fischer policy checkpoint override.
        output_dir: Optional directory for generated PGN and CSV artifacts.

    Returns:
        Arena score summary and paths to its generated records.

    Raises:
        FileNotFoundError: If the configured Stockfish executable is absent.
        ValueError: If numeric arena settings are invalid.
    """
    if games <= 0 or stockfish_time_seconds <= 0 or max_plies <= 0:
        raise ValueError("games, stockfish_time_seconds, and max_plies must be positive.")

    selected_stockfish_path = stockfish_path or Path(config.stockfish_path)
    selected_checkpoint_path = checkpoint_path or config.model_checkpoint_path
    selected_output_dir = output_dir or config.arena_output_dir
    if not selected_stockfish_path.is_file():
        raise FileNotFoundError(
            f"Stockfish executable does not exist: {selected_stockfish_path}"
        )

    fischer_ai = FischerAI(selected_checkpoint_path)
    opening_order = list(_OPENINGS)
    random.Random(seed).shuffle(opening_order)
    game_results: list[ArenaGameResult] = []
    pgn_games: list[chess.pgn.Game] = []

    with chess.engine.SimpleEngine.popen_uci(selected_stockfish_path) as engine:
        effective_elo = _configure_limited_stockfish(engine, requested_stockfish_elo)
        for game_number in tqdm(range(1, games + 1), desc="Stockfish arena"):
            opening_name, opening_moves = opening_order[(game_number - 1) % len(opening_order)]
            model_color = chess.WHITE if game_number % 2 == 1 else chess.BLACK
            result, game = _play_game(
                game_number,
                fischer_ai,
                engine,
                model_color,
                opening_name,
                opening_moves,
                stockfish_time_seconds,
                max_plies,
                effective_elo,
            )
            game_results.append(result)
            pgn_games.append(game)

    wins = sum(result.model_score == 1.0 for result in game_results)
    draws = sum(result.model_score == 0.5 for result in game_results)
    losses = sum(result.model_score == 0.0 for result in game_results)
    run_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pgn_path, csv_path = _write_artifacts(
        selected_output_dir,
        run_label,
        game_results,
        pgn_games,
    )
    return ArenaSummary(
        games=games,
        wins=wins,
        draws=draws,
        losses=losses,
        score=(wins + 0.5 * draws) / games,
        effective_stockfish_elo=effective_elo,
        pgn_path=pgn_path,
        csv_path=csv_path,
    )


def _parse_arguments() -> argparse.Namespace:
    """Parse command-line overrides for the configured arena defaults."""
    parser = argparse.ArgumentParser(
        description="Evaluate FischerAI against Elo-limited Stockfish."
    )
    parser.add_argument("--games", type=int, default=settings.arena_games)
    parser.add_argument("--elo", type=int, default=settings.arena_stockfish_elo)
    parser.add_argument(
        "--stockfish-move-time",
        type=float,
        default=settings.arena_stockfish_move_time_seconds,
    )
    parser.add_argument("--max-plies", type=int, default=settings.arena_max_plies)
    parser.add_argument("--seed", type=int, default=settings.arena_seed)
    parser.add_argument("--stockfish-path", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the configured Stockfish arena and print its aggregate score."""
    arguments = _parse_arguments()
    summary = run_stockfish_arena(
        config=settings,
        games=arguments.games,
        requested_stockfish_elo=arguments.elo,
        stockfish_time_seconds=arguments.stockfish_move_time,
        max_plies=arguments.max_plies,
        seed=arguments.seed,
        stockfish_path=arguments.stockfish_path,
        checkpoint_path=arguments.checkpoint_path,
        output_dir=arguments.output_dir,
    )
    print(f"Stockfish Elo: {summary.effective_stockfish_elo}")
    print(f"Games: {summary.games}")
    print(f"FischerAI W/D/L: {summary.wins}/{summary.draws}/{summary.losses}")
    print(f"FischerAI score: {summary.score:.2%}")
    print(f"PGN: {summary.pgn_path}")
    print(f"CSV: {summary.csv_path}")


if __name__ == "__main__":
    main()
