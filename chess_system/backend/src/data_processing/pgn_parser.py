"""
PGN parsing and per-move quality filtering for imitation-learning data.

Pipeline: PGN file -> games (streamed, never fully loaded into memory) ->
for each game, only the target player's own moves are considered -> each
candidate move is scored against Stockfish before and after being played;
moves that lose too much centipawn value ("blunders") are dropped.

This module only produces a *stream* of (fen, move) pairs — see
`write_training_examples_jsonl` for materializing that stream to a compact
on-disk cache once, ahead of actual model training (running Stockfish for
every move is too slow to repeat every epoch).
"""

from __future__ import annotations

import json
import logging
import re
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union, List

import chess
import chess.engine
import chess.pgn

logger = logging.getLogger(__name__)

BLUNDER_THRESHOLD_CP_DEFAULT = 150
MIN_CLASSICAL_BASE_SECONDS_DEFAULT = 900  # 15 min base time; excludes blitz/rapid/bullet
MATE_SCORE_CP = 100_000  # finite stand-in for mate scores, so cp subtraction stays well-defined

_TIME_CONTROL_BASE_RE = re.compile(r"^(\d+)")

def _game_belongs_to_worker(game_id: int, worker_id: int, num_workers: int) -> bool:
    """Deterministic game->worker assignment shared by sharding and its test."""
    return (game_id - 1) % num_workers == worker_id


@dataclass(frozen=True)
class TrainingExample:
    """One (state, action) pair for imitation learning, plus its source game."""

    fen: str
    move_uci: str
    game_id: int


def _player_color_in_game(headers: "chess.pgn.Headers", player_name: str) -> Optional[chess.Color]:
    """Return which color `player_name` played in this game, or None if neither side matches."""
    white = headers.get("White", "")
    black = headers.get("Black", "")
    needle = player_name.lower()
    if needle in white.lower():
        return chess.WHITE
    if needle in black.lower():
        return chess.BLACK
    return None


def _is_classical_time_control(headers: "chess.pgn.Headers", min_base_seconds: int) -> bool:
    """
    Best-effort classification of a game as "classical" from its headers.

    Historical archives (including essentially all of Fischer's PGNs) very
    often omit the `TimeControl` tag entirely — those are assumed classical
    by default. Where the tag IS present, we require a base time of at
    least `min_base_seconds`, which excludes blitz/rapid/bullet games.
    """
    time_control = headers.get("TimeControl", "")
    if not time_control or time_control == "-":
        return True

    match = _TIME_CONTROL_BASE_RE.match(time_control)
    if not match:
        return True  # unparseable format — don't reject on a guess

    base_seconds = int(match.group(1))
    return base_seconds >= min_base_seconds


def _process_game(
    game: "chess.pgn.Game",
    engine: "chess.engine.SimpleEngine",
    game_id: int,
    player_name: str,
    depth: int,
    blunder_threshold_cp: int,
    min_classical_base_seconds: int,
) -> Iterator[TrainingExample]:
    """Yield filtered (fen, move) pairs for one game's target-player moves."""
    headers = game.headers

    if not _is_classical_time_control(headers, min_classical_base_seconds):
        logger.info(
            "Game #%d skipped: non-classical time control (TimeControl=%r).",
            game_id, headers.get("TimeControl"),
        )
        return

    target_color = _player_color_in_game(headers, player_name)
    if target_color is None:
        logger.info("Game #%d skipped: '%s' not found in White/Black headers.", game_id, player_name)
        return

    board = game.board()
    limit = chess.engine.Limit(depth=depth)


    try:
        engine.configure({"Clear Hash": None})
    except Exception:
        logger.debug("Game #%d: engine does not support 'Clear Hash' — continuing without it.", game_id)

    for move in game.mainline_moves():
        # Explicit legality check before doing anything else with this move —
        # protects against corrupted/hand-edited PGNs where python-chess's
        # own parser may not have caught an inconsistency.
        if move not in board.legal_moves:
            logger.warning(
                "Game #%d: illegal move %s at FEN %r — stopping parse of this game.",
                game_id, move.uci(), board.fen(),
            )
            return

        mover_color = board.turn

        if mover_color != target_color:
            # Opponent's move: advance the board so future positions stay
            # correct, but don't spend engine time on it or yield it.
            board.push(move)
            continue

        fen_before = board.fen()

        try:
            score_before = (
                engine.analyse(board, limit)["score"].pov(mover_color).score(mate_score=MATE_SCORE_CP)
            )
        except Exception:
            logger.warning(
                "Game #%d: Stockfish analysis failed before move %s — skipping this move.",
                game_id, move.uci(), exc_info=True,
            )
            board.push(move)
            continue

        board.push(move)

        try:
            score_after = (
                engine.analyse(board, limit)["score"].pov(mover_color).score(mate_score=MATE_SCORE_CP)
            )
        except Exception:
            logger.warning(
                "Game #%d: Stockfish analysis failed after move %s — skipping this move.",
                game_id, move.uci(), exc_info=True,
            )
            continue

        delta_cp = score_after - score_before
        if delta_cp < -blunder_threshold_cp:
            logger.debug(
                "Game #%d: dropping blunder %s (Delta cp=%d) at FEN %r.",
                game_id, move.uci(), delta_cp, fen_before,
            )
            continue

        yield TrainingExample(fen=fen_before, move_uci=move.uci(), game_id=game_id)


def stream_training_examples(
    pgn_path: Union[str, Path],
    stockfish_path: str,
    *,
    player_name: str = "Fischer",
    depth: int = 8,
    blunder_threshold_cp: int = BLUNDER_THRESHOLD_CP_DEFAULT,
    min_classical_base_seconds: int = MIN_CLASSICAL_BASE_SECONDS_DEFAULT,
    max_games: Optional[int] = None,
) -> Iterator[TrainingExample]:
    """
    Stream (fen, move) training pairs from a PGN file, filtered for quality.

    Filtering applied, in order:
        1. Time control: games without a `TimeControl` header are assumed
           classical; games with one are kept only if base time >=
           `min_classical_base_seconds`.
        2. Player match: the game is kept only if `player_name` appears in
           the White or Black header; ONLY that player's own moves are
           yielded (opponent moves update the board but are not emitted —
           we're building a dataset of the target player's decisions, not
           of "any move played against them").
        3. Move-level quality: every candidate move is evaluated with
           Stockfish immediately before and after being played (from the
           mover's point of view). If centipawn score drops by more than
           `blunder_threshold_cp` (default 150), the move is dropped as a
           blunder — win/loss/draw games are ALL used (we don't filter by
           game result), so the model can still learn defensive play from
           games Fischer eventually lost, minus the actual losing blunder.
        4. Legality: `move in board.legal_moves` is checked immediately
           before processing, to guarantee every yielded pair is playable.

    Corrupt PGN entries, unparseable individual games, and Stockfish
    analysis failures are logged as warnings and skipped — this function
    never raises for bad *data*. It WILL raise if `stockfish_path` doesn't
    point at a working UCI engine (a configuration error, not a data
    error), since that should fail loudly rather than silently produce an
    empty dataset.

    Args:
        pgn_path: Path to a (potentially very large) PGN file.
        stockfish_path: Path to a UCI-compatible Stockfish binary.
        player_name: Case-insensitive substring to match against the
            White/Black PGN headers; only this player's moves are yielded.
        depth: Stockfish search depth used for both before/after
            evaluations. Kept low by default for throughput.
        blunder_threshold_cp: Centipawn-loss threshold beyond which a move
            is dropped (positive integer; compared against -delta_cp).
        min_classical_base_seconds: Minimum PGN `TimeControl` base time (in
            seconds) to count as classical, when that header is present.
        max_games: Optional cap on the number of games read from the file
            (mainly useful for smoke-testing a pipeline run).

    Yields:
        TrainingExample(fen, move_uci, game_id) for each surviving move.
    """
    pgn_path = Path(pgn_path)

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as pgn_file, \
            chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:

        game_id = 0
        while True:
            if max_games is not None and game_id >= max_games:
                break

            try:
                game = chess.pgn.read_game(pgn_file)
            except Exception:
                logger.warning(
                    "Failed to parse a game from %s (corrupt PGN entry) — skipping.",
                    pgn_path, exc_info=True,
                )
                continue

            if game is None:
                break  # end of file

            game_id += 1

            try:
                yield from _process_game(
                    game=game,
                    engine=engine,
                    game_id=game_id,
                    player_name=player_name,
                    depth=depth,
                    blunder_threshold_cp=blunder_threshold_cp,
                    min_classical_base_seconds=min_classical_base_seconds,
                )
            except Exception:
                logger.warning(
                    "Unexpected error while processing game #%d of %s — skipping.",
                    game_id, pgn_path, exc_info=True,
                )
                continue

def _process_pgn_shard(
    pgn_path: str,
    stockfish_path: str,
    worker_id: int,
    num_workers: int,
    player_name: str,
    depth: int,
    blunder_threshold_cp: int,
    min_classical_base_seconds: int,
    max_games: Optional[int],
    log_level: int,
) -> List[TrainingExample]:
    """
    Runs in its own OS process (via ProcessPoolExecutor): opens its own PGN
    file handle and its own Stockfish engine, and processes only the games
    assigned to `worker_id` (see `_game_belongs_to_worker`).

    Every worker re-reads the whole PGN file sequentially and skips games
    that aren't its own — cheap, since PGN text parsing is negligible next
    to a single Stockfish call. This avoids needing byte-offset bookkeeping
    to give each worker a physical slice of the file.

    Must stay a module-level function (not a closure/lambda): on Windows,
    `ProcessPoolExecutor` uses the "spawn" start method, which pickles the
    target function by import path — closures aren't picklable that way.

    Returns a plain list (not a generator) since results cross a process
    boundary via pickling, which requires a fully materialized value.
    """
    logging.basicConfig(level=log_level)  # spawned processes don't inherit the parent's logging config

    results: List[TrainingExample] = []
    pgn_path_obj = Path(pgn_path)

    with open(pgn_path_obj, "r", encoding="utf-8", errors="replace") as pgn_file, \
            chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:

        game_id = 0
        while True:
            if max_games is not None and game_id >= max_games:
                break

            try:
                game = chess.pgn.read_game(pgn_file)
            except Exception:
                logger.warning(
                    "[worker %d] Failed to parse a game from %s — skipping.",
                    worker_id, pgn_path_obj, exc_info=True,
                )
                continue

            if game is None:
                break  # end of file

            game_id += 1

            if not _game_belongs_to_worker(game_id, worker_id, num_workers):
                continue  # another worker owns this game

            try:
                results.extend(_process_game(
                    game=game,
                    engine=engine,
                    game_id=game_id,
                    player_name=player_name,
                    depth=depth,
                    blunder_threshold_cp=blunder_threshold_cp,
                    min_classical_base_seconds=min_classical_base_seconds,
                ))
            except Exception:
                logger.warning(
                    "[worker %d] Unexpected error while processing game #%d of %s — skipping.",
                    worker_id, game_id, pgn_path_obj, exc_info=True,
                )
                continue

    logger.info("[worker %d] done: %d examples from its shard.", worker_id, len(results))
    return results


def stream_training_examples_parallel(
    pgn_path: Union[str, Path],
    stockfish_path: str,
    *,
    num_workers: Optional[int] = None,
    player_name: str = "Fischer",
    depth: int = 8,
    blunder_threshold_cp: int = BLUNDER_THRESHOLD_CP_DEFAULT,
    min_classical_base_seconds: int = MIN_CLASSICAL_BASE_SECONDS_DEFAULT,
    max_games: Optional[int] = None,
) -> Iterator[TrainingExample]:
    """
    Same filtering behavior as `stream_training_examples`, parallelized
    across games using a process per worker (each with its own Stockfish
    instance) — this is where the real speedup is: games are independent
    of each other, even though moves *within* a game must stay sequential.

    Trade-off vs. `stream_training_examples`: this is NOT constant-memory
    streaming. Each worker's shard is collected into a list before
    returning (`ProcessPoolExecutor` results must be picklable, which rules
    out yielding across the process boundary), and this function then
    yields from the combined, game_id-sorted results. At this dataset's
    scale (hundreds of games, tens of thousands of short text examples)
    that's a trivial amount of memory — for a much larger corpus, a
    streaming producer/consumer design (e.g. a `multiprocessing.Queue`)
    would be worth the added complexity; not needed here.

    `num_workers <= 1` (including the default of `None`, which resolves to
    `os.cpu_count()`) falls back to the plain single-process
    `stream_training_examples` — no process pool overhead for small runs.

    Windows note: `ProcessPoolExecutor` uses the "spawn" start method there,
    which re-imports your entry-point module in each worker. Call this from
    inside `if __name__ == "__main__":` (already true of
    `src/data_processing/build_cache.py`) or worker processes will recurse.

    Args: identical to `stream_training_examples`, plus:
        num_workers: Number of worker processes. Defaults to
            `os.cpu_count()`. Each worker opens its own Stockfish process,
            so this should not exceed available CPU cores.

    Yields:
        TrainingExample(fen, move_uci, game_id), ordered by game_id (stable
        across runs, unlike raw worker-completion order).
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    if num_workers <= 1:
        yield from stream_training_examples(
            pgn_path,
            stockfish_path,
            player_name=player_name,
            depth=depth,
            blunder_threshold_cp=blunder_threshold_cp,
            min_classical_base_seconds=min_classical_base_seconds,
            max_games=max_games,
        )
        return

    pgn_path = str(pgn_path)
    log_level = logger.getEffectiveLevel()

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(
                _process_pgn_shard,
                pgn_path,
                stockfish_path,
                worker_id,
                num_workers,
                player_name,
                depth,
                blunder_threshold_cp,
                min_classical_base_seconds,
                max_games,
                log_level,
            )
            for worker_id in range(num_workers)
        ]
        shards = [future.result() for future in futures]

    combined = [example for shard in shards for example in shard]
    combined.sort(key=lambda example: example.game_id)
    yield from combined

def write_training_examples_jsonl(
    examples: Iterable[TrainingExample],
    output_path: Union[str, Path],
) -> int:
    """
    Materialize a stream of TrainingExample into a JSONL cache file.

    This is the one place the (Stockfish-backed, expensive) filtering pass
    is meant to run — actual training epochs read this compact cache
    instead (see `data_processing.dataset.get_dataloaders`), so PGN
    re-parsing and engine analysis never repeat per-epoch.

    Args:
        examples: Typically the generator returned by
            `stream_training_examples`.
        output_path: Destination `.jsonl` file; parent directories are
            created if needed.

    Returns:
        Number of examples written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps({
                "fen": example.fen,
                "move_uci": example.move_uci,
                "game_id": example.game_id,
            }))
            f.write("\n")
            count += 1

    logger.info("Wrote %d training examples to %s", count, output_path)
    return count