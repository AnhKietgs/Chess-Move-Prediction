# data_processing

Data pipeline for the Behavioral Cloning training set (Bobby Fischer's
games). Three-stage pipeline, each stage in its own module:

## 1. `pgn_parser.py` — PGN → filtered (FEN, move) pairs

`stream_training_examples(pgn_path, stockfish_path, ...)` is a generator
that streams games from a PGN file one at a time (never loads the whole
file into memory) and yields `TrainingExample(fen, move_uci, game_id)`.

Filtering applied:
- **Classical time control only** — games with a `TimeControl` header
  below `min_classical_base_seconds` (default 900s) are skipped. Games
  with no `TimeControl` tag (true of most historical Fischer PGNs) are
  assumed classical.
- **Target player's own moves only** — only moves made by the player
  matching `player_name` (default `"Fischer"`) are yielded; the opponent's
  moves update the board but are never emitted as training examples.
- **Blunder filtering, not result filtering** — win/loss/draw games are
  ALL used. Each candidate move is scored with Stockfish immediately
  before and after being played; moves that lose more than
  `blunder_threshold_cp` (default 150) centipawns are dropped. This is
  how the pipeline avoids learning outright mistakes without also
  discarding entire games Fischer lost (which still contain plenty of
  sound moves worth imitating).
- **Legality** — `move in board.legal_moves` is re-checked immediately
  before yielding, independent of python-chess's own SAN parsing.
- **Reproducibility** — Stockfish's transposition hash table persists
  across `analyse()` calls within one engine process; left uncleared, the
  *same* game can yield a different blunder judgment depending purely on
  which other games were analysed by that engine beforehand. `Clear Hash`
  is sent at the start of every game so results never depend on
  processing order (this is also what makes the parallel path below give
  byte-identical output to the sequential one).

Corrupt PGN entries and per-move Stockfish failures are logged and
skipped; a bad `stockfish_path` (configuration error) raises immediately.

`write_training_examples_jsonl(examples, output_path)` materializes the
generator's output to a compact `.jsonl` cache on disk — this is meant to
run ONCE, ahead of training, since re-running Stockfish analysis every
epoch would be far too slow.

### Parallel pipeline

`stream_training_examples_parallel(pgn_path, stockfish_path, num_workers=...)`
— same filtering, same output, but fans games out across `num_workers`
processes (each with its own Stockfish instance), since different games
are fully independent even though moves *within* one game must stay
sequential. `build_cache.py` uses this by default
(`num_workers=os.cpu_count()`); pass `--num-workers 1` to force the
original single-process path. Not constant-memory like the sequential
generator (each worker's shard is collected into a list before returning,
since results have to cross a process boundary) — a non-issue at this
dataset's scale (hundreds of games).

## 2. `encoder.py` — (FEN, move) → tensors

- `fen_to_tensor(fen) -> torch.Tensor`: `[18, 8, 8]`, `dtype=torch.float32`.
  12 piece planes + 4 castling-rights planes + 1 en-passant plane + 1
  side-to-move plane.
- `move_to_index(move) -> int` / `index_to_move(index, board) -> chess.Move`:
  AlphaZero-style 73-plane action space (4672 total classes), for use with
  `nn.CrossEntropyLoss`. Raises `ValueError` for any move that isn't a
  real straight-line/diagonal/knight-shaped displacement.

## 3. `dataset.py` — cache → DataLoaders

`get_dataloaders(file_path, batch_size, num_workers)` reads the JSONL
cache, splits it into train/val/test **at the game level** (via
`split_indices_by_game`) so consecutive, highly-correlated positions from
one game never leak across splits, and returns
`{"train": DataLoader, "val": DataLoader, "test": DataLoader}`. Tensor
encoding happens lazily inside `ChessDataset.__getitem__`, so it runs in
parallel across `num_workers` DataLoader workers instead of up front.

## Typical usage

The `Fischer.pgn` dataset (827 games) is bundled at `backend/data/raw/Fischer.pgn`
— see `backend/data/README.md` for the full walkthrough. Quick version:

```bash
cd backend
python -m src.data_processing.build_cache   # writes data/cache/fischer_training_examples.jsonl
```

```python
from src.data_processing.dataset import get_dataloaders

loaders = get_dataloaders("data/cache/fischer_training_examples.jsonl", batch_size=256, num_workers=4)
for board_tensor, move_label in loaders["train"]:
    ...  # board_tensor: [B, 18, 8, 8] float32, move_label: [B] long
```

Tests: `backend/tests/test_data_pipeline.py` (encoder shape/dtype/error
cases, plus leakage-free-split checks — run with `pytest` from `backend/`).
