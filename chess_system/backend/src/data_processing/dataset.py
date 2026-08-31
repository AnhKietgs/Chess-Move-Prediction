"""
PyTorch Dataset, leakage-safe train/val/test split, and DataLoader
construction for the Behavioral Cloning chess policy.

Reads from the compact JSONL cache produced by
`data_processing.pgn_parser.write_training_examples_jsonl` — NOT the raw
PGN/Stockfish pipeline directly, so the expensive filtering pass only ever
runs once. Encoding to tensors happens lazily, per-sample, inside
`ChessDataset.__getitem__`, so a full epoch never materializes the whole
dataset as tensors in memory at once — combined with `num_workers > 0` on
the returned DataLoaders, this keeps RAM bounded regardless of dataset size.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import chess
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.data_processing.encoder import (
    ACTION_SPACE_SIZE,
    board_to_tensor,
    mirror_fen,
    mirror_move,
    move_to_index,
)

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Record:
    """Lightweight (string-only) row loaded from the JSONL cache."""

    fen: str
    move_uci: str
    game_id: int


@dataclass(frozen=True)
class FenDisjointSplitAudit:
    """Report how strict FEN filtering changed a game-level split.

    Attributes:
        train_examples: Number of training examples, never removed by this
            evaluation-only filtering step.
        validation_examples: Number of retained validation examples.
        test_examples: Number of retained test examples.
        validation_removed: Validation examples removed because their
            normalized FEN appeared in training.
        test_removed: Test examples removed because their normalized FEN
            appeared in training or validation.
        train_unique_fens: Number of unique normalized FENs in training.
        validation_unique_fens: Number of unique normalized FENs retained in
            validation.
        test_unique_fens: Number of unique normalized FENs retained in test.
    """

    train_examples: int
    validation_examples: int
    test_examples: int
    validation_removed: int
    test_removed: int
    train_unique_fens: int
    validation_unique_fens: int
    test_unique_fens: int


def normalized_fen_key(fen: str) -> str:
    """Return the rule-relevant, four-field identity key for a FEN.

    Halfmove and fullmove counters do not alter legal moves or board state, so
    they are deliberately excluded. The remaining fields encode pieces, side
    to move, castling rights, and en-passant target.

    Args:
        fen: Full Forsyth-Edwards Notation string.

    Returns:
        The first four normalized FEN fields joined by spaces.

    Raises:
        ValueError: If ``fen`` does not contain at least four fields.
    """
    fields = fen.split()
    if len(fields) < 4:
        raise ValueError(f"FEN must contain at least four fields: {fen!r}")
    return " ".join(fields[:4])


def _load_records(cache_path: Union[str, Path]) -> List[_Record]:
    """Load the JSONL cache into a list of `_Record`s.

    Deliberately kept to strings/ints only (no tensors) — for a
    multi-million-position dataset this list is still only tens to
    hundreds of MB, comfortably fitting in RAM, while the actual tensors
    (which would NOT fit) are built lazily per-sample.
    """
    cache_path = Path(cache_path)
    records: List[_Record] = []

    with open(cache_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                records.append(
                    _Record(
                        fen=payload["fen"],
                        move_uci=payload["move_uci"],
                        game_id=payload["game_id"],
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                    f"Malformed cache entry at line {line_number} of {cache_path}: {exc}"
                ) from exc

    if not records:
        raise ValueError(f"No training examples found in cache file {cache_path}.")

    return records


class ChessDataset(Dataset):
    """Lazily encode board, target action, and legal-action mask per sample.

    Built from a list of `_Record`s rather than a file path directly, so
    `get_dataloaders` can load the cache once and share the same records
    across the train/val/test `Subset`s without re-reading the file. The
    legal-action mask is generated on the CPU in ``__getitem__``; DataLoader
    workers therefore parallelize python-chess move generation rather than
    performing it in the GPU training loop.
    """

    def __init__(self, records: Sequence[_Record], augment: bool = False):
        """Initialize a lazy dataset with optional horizontal mirroring.

        Args:
            records: Cache records shared by one or more dataset splits.
            augment: When true, each access has a 50% probability of
                horizontally mirroring state, target, and legal actions.
        """
        self._records = records
        self._augment = augment

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return encoded board state, target action, and legal action mask.

        Args:
            idx: Index into the dataset records.

        Returns:
            A tuple containing a board tensor ``[18, 8, 8]``, a scalar
            ``torch.long`` action label, and a boolean legal-move mask of
            shape ``[4672]``.

        Raises:
            ValueError: If the cached target move is illegal for its FEN.
        """
        record = self._records[idx]
        source_board = chess.Board(record.fen)
        source_move = chess.Move.from_uci(record.move_uci)
        source_legal_moves = list(source_board.legal_moves)

        # Horizontal reflection is an exact chess symmetry only after both
        # sides have lost castling rights. Reflecting an orthodox castling
        # move (for example e1g1 -> d1b1) is not a legal orthodox castling
        # move, so augmenting those positions would create inconsistent
        # board/label pairs.
        can_mirror = not source_board.has_castling_rights(
            chess.WHITE
        ) and not source_board.has_castling_rights(chess.BLACK)
        if self._augment and can_mirror and random.random() < 0.5:
            mirrored_fen = mirror_fen(record.fen)
            board = chess.Board(mirrored_fen)
            move = mirror_move(source_move)
            legal_moves = list(board.legal_moves)
            if move not in legal_moves:
                raise ValueError(
                    "Mirrored target move is illegal for its mirrored training FEN: "
                    f"{record.fen}"
                )
        else:
            board = source_board
            move = source_move
            legal_moves = source_legal_moves

        board_tensor = board_to_tensor(board)
        label = torch.tensor(move_to_index(move), dtype=torch.long)
        legal_move_mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
        legal_indices = [move_to_index(legal_move) for legal_move in legal_moves]
        legal_move_mask[legal_indices] = True
        if not legal_move_mask[label.item()]:
            raise ValueError(
                f"Cached move {record.move_uci} is illegal for training FEN: {record.fen}"
            )
        return board_tensor, label, legal_move_mask


def split_indices_by_game(
    records: Sequence[_Record],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Split dataset indices into train/val/test WITHOUT data leakage.

    Positions from the same game are highly correlated (consecutive board
    states differ by one move), so splitting at the *position* level would
    let near-duplicate positions from one game leak across splits and
    inflate validation/test scores. Instead, whole `game_id`s are shuffled
    and split, and every position belonging to a given game is assigned to
    that game's split.

    Args:
        records: All loaded records, as returned by `_load_records`.
        train_ratio, val_ratio, test_ratio: Must sum to 1.0.
        seed: RNG seed for the game-id shuffle (reproducible splits).

    Returns:
        {"train": [...], "val": [...], "test": [...]}: lists of indices
        into `records`.

    Raises:
        ValueError: If the ratios don't sum to 1.0.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    game_to_indices: Dict[int, List[int]] = {}
    for idx, record in enumerate(records):
        game_to_indices.setdefault(record.game_id, []).append(idx)

    game_ids = list(game_to_indices.keys())
    random.Random(seed).shuffle(game_ids)

    n_games = len(game_ids)
    # Plain `round()` uses round-half-to-even (round(0.5) == 0), which can
    # silently zero out the smaller split on small datasets. Round-half-up
    # instead, so e.g. 5 games at a 0.1 val_ratio gives 1 game, not 0.
    n_train = math.floor(n_games * train_ratio + 0.5)
    n_val = math.floor(n_games * val_ratio + 0.5)

    train_game_ids = set(game_ids[:n_train])
    val_game_ids = set(game_ids[n_train:n_train + n_val])
    # Remainder goes to test — avoids rounding drift leaving games unassigned.

    splits: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for game_id, indices in game_to_indices.items():
        if game_id in train_game_ids:
            splits["train"].extend(indices)
        elif game_id in val_game_ids:
            splits["val"].extend(indices)
        else:
            splits["test"].extend(indices)

    return splits


def enforce_fen_disjoint_splits(
    records: Sequence[_Record],
    splits: Mapping[str, Sequence[int]],
) -> tuple[Dict[str, List[int]], FenDisjointSplitAudit]:
    """Remove exact-position overlap from validation and test splits.

    The input split must already be grouped by game. Training examples are
    retained unchanged. Validation loses positions seen in training; test
    then loses positions seen in either training or retained validation. This
    keeps every held-out FEN genuinely unseen without moving individual game
    positions into the training set.

    Args:
        records: Cache records addressed by the split indices.
        splits: Existing ``train``, ``val``, and ``test`` index collections.

    Returns:
        FEN-disjoint split indices and an audit report describing removals.

    Raises:
        KeyError: If a required split name is missing.
        IndexError: If an index is outside ``records``.
    """
    required_splits = ("train", "val", "test")
    missing_splits = [name for name in required_splits if name not in splits]
    if missing_splits:
        raise KeyError(f"Missing required split(s): {missing_splits}")

    disjoint_splits: Dict[str, List[int]] = {
        "train": list(splits["train"]),
        "val": [],
        "test": [],
    }
    seen_fens = {
        normalized_fen_key(records[index].fen)
        for index in disjoint_splits["train"]
    }
    train_unique_fens = len(seen_fens)

    validation_removed = 0
    for index in splits["val"]:
        fen_key = normalized_fen_key(records[index].fen)
        if fen_key in seen_fens:
            validation_removed += 1
            continue
        disjoint_splits["val"].append(index)
        seen_fens.add(fen_key)
    validation_unique_fens = len(
        {normalized_fen_key(records[index].fen) for index in disjoint_splits["val"]}
    )

    test_removed = 0
    for index in splits["test"]:
        fen_key = normalized_fen_key(records[index].fen)
        if fen_key in seen_fens:
            test_removed += 1
            continue
        disjoint_splits["test"].append(index)
        seen_fens.add(fen_key)
    test_unique_fens = len(
        {normalized_fen_key(records[index].fen) for index in disjoint_splits["test"]}
    )

    return disjoint_splits, FenDisjointSplitAudit(
        train_examples=len(disjoint_splits["train"]),
        validation_examples=len(disjoint_splits["val"]),
        test_examples=len(disjoint_splits["test"]),
        validation_removed=validation_removed,
        test_removed=test_removed,
        train_unique_fens=train_unique_fens,
        validation_unique_fens=validation_unique_fens,
        test_unique_fens=test_unique_fens,
    )


def get_dataloaders(
    file_path: Union[str, Path],
    batch_size: int = 256,
    num_workers: int = 4,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
    augment_train: bool = True,
    strict_fen_disjoint: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders from a JSONL training-examples cache.

    Args:
        file_path: Path to the JSONL cache written by
            `pgn_parser.write_training_examples_jsonl`.
        batch_size: Batch size used for all three loaders.
        num_workers: Passed to each DataLoader for multiprocessed, parallel
            FEN-to-tensor encoding (see module docstring).
        train_ratio, val_ratio, test_ratio: Split proportions. Split is
            game-level, not leaked — see `split_indices_by_game`.
        seed: RNG seed for the split.
        augment_train: Whether to apply random horizontal mirroring to the
            training split only. Validation and test samples are never
            augmented.
        strict_fen_disjoint: When true, remove validation/test examples whose
            normalized FEN is already present in an earlier split. This gives
            an exact-position-disjoint evaluation while preserving game-level
            separation.

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    records = _load_records(file_path)
    splits = split_indices_by_game(records, train_ratio, val_ratio, test_ratio, seed)
    if strict_fen_disjoint:
        splits, audit = enforce_fen_disjoint_splits(records, splits)
        logger.info(
            "Strict FEN-disjoint split: train=%d (%d FENs), val=%d "
            "(removed=%d; %d FENs), test=%d (removed=%d; %d FENs).",
            audit.train_examples,
            audit.train_unique_fens,
            audit.validation_examples,
            audit.validation_removed,
            audit.validation_unique_fens,
            audit.test_examples,
            audit.test_removed,
            audit.test_unique_fens,
        )

    loaders: Dict[str, DataLoader] = {}
    for split_name, indices in splits.items():
        dataset = ChessDataset(
            records,
            augment=augment_train and split_name == "train",
        )
        subset = Subset(dataset, indices)
        loaders[split_name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    return loaders
