"""
CLI entrypoint: run the full (slow, Stockfish-backed) pipeline once and
write the resulting cache to disk. Meant to be run manually, ahead of
training — not imported by application code.

Usage:
    python -m src.data_processing.build_cache
    python -m src.data_processing.build_cache --depth 6 --max-games 50   # quick smoke test
"""

from __future__ import annotations

import argparse
import os
import logging
import shutil
import sys
from pathlib import Path

from src.data_processing.pgn_parser import (
    BLUNDER_THRESHOLD_CP_DEFAULT,
    stream_training_examples_parallel,
    write_training_examples_jsonl,
)

DEFAULT_PGN_PATH = "data/raw/Fischer.pgn"
DEFAULT_CACHE_PATH = "data/cache/fischer_training_examples.jsonl"
DEFAULT_DEPTH = 8


_COMMON_STOCKFISH_PATHS = (
    "/usr/games/stockfish",  # Debian/Ubuntu apt package — not always on $PATH
    "/usr/local/bin/stockfish",
    "/usr/bin/stockfish",
    "/opt/homebrew/bin/stockfish",  # macOS, Apple Silicon Homebrew
)


def _resolve_stockfish_path(explicit: str | None) -> str:
    """Prefer an explicit --stockfish-path, then settings/$STOCKFISH_PATH, then PATH, then common install locations."""
    if explicit:
        return explicit

    try:
        from src.config.settings import settings

        if settings.stockfish_path and Path(settings.stockfish_path).exists():
            return settings.stockfish_path
    except Exception:
        pass  # settings module optional in this context; fall through to PATH lookup

    found = shutil.which("stockfish")
    if found:
        return found

    for candidate in _COMMON_STOCKFISH_PATHS:
        if Path(candidate).exists():
            return candidate

    raise SystemExit(
        "Could not locate a Stockfish binary. Pass --stockfish-path explicitly, "
        "set STOCKFISH_PATH in backend/.env, or install it (e.g. `apt-get install stockfish`)."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn-path", default=DEFAULT_PGN_PATH)
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--stockfish-path", default=None)
    parser.add_argument("--player-name", default="Fischer")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--blunder-threshold-cp", type=int, default=BLUNDER_THRESHOLD_CP_DEFAULT)
    parser.add_argument(
        "--max-games", type=int, default=None, help="Cap games read (useful for a quick smoke test)."
    )
    parser.add_argument(
            "--num-workers",
            type=int,
            default=None,
            help="Worker processes, each with its own Stockfish instance (default: os.cpu_count()). "
                 "Use 1 to force the original single-process path.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    stockfish_path = _resolve_stockfish_path(args.stockfish_path)
    num_workers = args.num_workers or (os.cpu_count() or 1)

    examples =  stream_training_examples_parallel(
        args.pgn_path,
        stockfish_path,
        num_workers=num_workers,
        player_name=args.player_name,
        depth=args.depth,
        blunder_threshold_cp=args.blunder_threshold_cp,
        max_games=args.max_games,
    )
    count = write_training_examples_jsonl(examples, args.cache_path)
    print(f"Done: {count} training examples written to {args.cache_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
