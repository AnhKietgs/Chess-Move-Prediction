# data

Not part of `src/` on purpose — this holds actual data files, not code.

```
data/
├── raw/     # source PGN files go here (e.g. Fischer.pgn)
└── cache/   # .jsonl output of the Stockfish-filtered pipeline (generated, not hand-edited)
```

`Fischer.pgn` (827 games, all with Fischer as either White or Black) is
already in `raw/`.

## Running the full pipeline

This runs Stockfish on every one of Fischer's own moves across all 827
games — expect it to take a while (each position is a real engine call).
Run it once; the result is cached to `data/cache/` and re-used for every
future training run.

```bash
cd backend
python -m src.data_processing.build_cache
```

Or from a Python shell / notebook, with more control over the knobs:

```python
from src.data_processing.pgn_parser import stream_training_examples, write_training_examples_jsonl

examples = stream_training_examples(
    "data/raw/Fischer.pgn",
    "/usr/games/stockfish",   # wherever Stockfish is installed — see below
    player_name="Fischer",
    depth=8,                  # lower (e.g. 6) for a faster, slightly noisier pass
    blunder_threshold_cp=150,
)
write_training_examples_jsonl(examples, "data/cache/fischer_training_examples.jsonl")
```

Then build DataLoaders from the cache:

```python
from src.data_processing.dataset import get_dataloaders

loaders = get_dataloaders("data/cache/fischer_training_examples.jsonl", batch_size=256, num_workers=4)
```

## Installing Stockfish

Not a Python package — install the binary separately.

- **Ubuntu/Debian**: `sudo apt-get install stockfish` → typically lands at `/usr/games/stockfish`
- **macOS**: `brew install stockfish` → typically `/opt/homebrew/bin/stockfish`
- **Windows**: download from https://stockfishchess.org/download/ and point at the `.exe` path

Whatever the path turns out to be, put it in `backend/.env` as `STOCKFISH_PATH` (already a recognized setting — see `src/config/settings.py`) rather than hardcoding it, so the training script and the API stay in sync.
