"""Tests for strict normalized-FEN split filtering."""

from src.data_processing.dataset import (
    _Record,
    enforce_fen_disjoint_splits,
    normalized_fen_key,
)


def test_normalized_fen_key_ignores_move_counters() -> None:
    """Move clocks do not make equivalent positions distinct."""
    first_fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
    equivalent_fen = "8/8/8/8/8/8/8/K6k w - - 37 99"

    assert normalized_fen_key(first_fen) == normalized_fen_key(equivalent_fen)


def test_enforce_fen_disjoint_splits_removes_cross_split_positions() -> None:
    """Validation/test positions already seen in earlier splits are removed."""
    shared_fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
    train_only_fen = "8/8/8/8/8/8/8/K6k b - - 0 1"
    validation_only_fen = "8/8/8/8/8/8/7k/K7 w - - 0 1"
    test_only_fen = "8/8/8/8/8/8/7k/K7 b - - 0 1"
    records = [
        _Record(shared_fen, "a1a2", 1),
        _Record(train_only_fen, "h1h2", 1),
        _Record(shared_fen.replace("0 1", "10 7"), "a1a2", 2),
        _Record(validation_only_fen, "a1a2", 2),
        _Record(validation_only_fen.replace("0 1", "4 3"), "h2h1", 3),
        _Record(test_only_fen, "a1a2", 3),
    ]
    splits = {"train": [0, 1], "val": [2, 3], "test": [4, 5]}

    strict_splits, audit = enforce_fen_disjoint_splits(records, splits)

    assert strict_splits == {"train": [0, 1], "val": [3], "test": [5]}
    assert audit.validation_removed == 1
    assert audit.test_removed == 1

    train_fens = {normalized_fen_key(records[index].fen) for index in strict_splits["train"]}
    validation_fens = {normalized_fen_key(records[index].fen) for index in strict_splits["val"]}
    test_fens = {normalized_fen_key(records[index].fen) for index in strict_splits["test"]}
    assert train_fens.isdisjoint(validation_fens)
    assert train_fens.isdisjoint(test_fens)
    assert validation_fens.isdisjoint(test_fens)
