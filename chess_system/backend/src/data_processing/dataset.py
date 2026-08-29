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
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

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


@dataclass(frozen=True)
class _Record:
    """Lightweight (string-only) row loaded from the JSONL cache."""

    fen: str
    move_uci: str
    game_id: int


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

        if self._augment and random.random() < 0.5:
            mirrored_fen = mirror_fen(record.fen)
            # Horizontal reflection moves the orthodox king from e-file to
            # d-file. Chess960 mode preserves the original castling-right
            # flags in this reflected coordinate system.
            board = chess.Board(mirrored_fen, chess960=True)
            move = mirror_move(source_move)
            legal_moves = [mirror_move(legal_move) for legal_move in source_legal_moves]
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


def get_dataloaders(
    file_path: Union[str, Path],
    batch_size: int = 256,
    num_workers: int = 4,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
    augment_train: bool = True,
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

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    records = _load_records(file_path)
    splits = split_indices_by_game(records, train_ratio, val_ratio, test_ratio, seed)

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
