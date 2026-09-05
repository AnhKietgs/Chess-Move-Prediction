# training

Reserved for Weeks 4-6 (Behavioral Cloning) and the later RL phase
(Style-constrained PPO with KL-Divergence regularization against the
frozen BC policy).

- `train_bc.py`: supervised training loop with validation Top-1/Top-3/Top-5 metrics.
- `evaluate_bc.py`: held-out test-set evaluation of the checkpoint configured
  in `settings.model_checkpoint_path` (or an explicit override).
- `evaluate_vs_stockfish.py`: reproducible model-versus-Stockfish arena.

Run test evaluation from `backend` after training:

```powershell
..\.venv\Scripts\python.exe -m src.training.evaluate_bc
```

Run a 100-game Stockfish arena after training:

```powershell
..\.venv\Scripts\python.exe -m src.training.evaluate_vs_stockfish --games 100 --elo 1350
```
