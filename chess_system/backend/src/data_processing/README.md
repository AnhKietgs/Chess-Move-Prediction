# data_processing

Reserved for Weeks 1-3 deliverables:

- PGN bulk parsing/cleaning for Bobby Fischer's games (chunked, generator-based
  to avoid loading entire PGN files into RAM).
- FEN -> Tensor encoding (12x8x8 piece planes + castling/en-passant/turn planes).
- PyTorch `Dataset` / `DataLoader` classes with `num_workers` for multiprocessing.

No code here yet — this API scaffold ships with the random-move placeholder
in `src/services/chess_engine.py`.
