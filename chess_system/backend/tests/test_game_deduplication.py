"""Tests for duplicate-PGN-game detection before cache generation."""

from pathlib import Path

from src.data_processing.pgn_parser import find_duplicate_game_ids


def test_find_duplicate_game_ids_ignores_headers_and_comments(tmp_path: Path) -> None:
    """Repeated main lines are detected even when PGN metadata differs."""
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(
        """[Event \"First\"]
[White \"Fischer\"]
[Black \"Opponent\"]

1. e4 e5 2. Nf3 Nc6 1/2-1/2

[Event \"Copied metadata changed\"]
[White \"Fischer\"]
[Black \"Opponent\"]

1. e4 {same game} e5 2. Nf3 Nc6 1/2-1/2

[Event \"Distinct game\"]
[White \"Fischer\"]
[Black \"Opponent\"]

1. d4 d5 1/2-1/2
""",
        encoding="utf-8",
    )

    assert find_duplicate_game_ids(pgn_path) == {2}
