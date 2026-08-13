# training

Reserved for Weeks 4-6 (Behavioral Cloning) and the later RL phase
(Style-constrained PPO with KL-Divergence regularization against the
frozen BC policy).

- `train_bc.py`: supervised training loop with validation Top-1/Top-3 metrics.
- `evaluate_bc.py`: held-out test-set evaluation of `best_fischer_bc.pth`.
- `train_rl.py` (future): self-play PPO loop with KL penalty term.

Run test evaluation from `backend` after training:

```powershell
..\.venv\Scripts\python.exe -m src.training.evaluate_bc
```
