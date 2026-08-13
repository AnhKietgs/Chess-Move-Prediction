"""Evaluate a trained Fischer Behavioral Cloning policy on the test split.

Run from the ``backend`` directory with ``python -m
src.training.evaluate_bc``. The script evaluates only the held-out test
split and never updates model parameters or checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config.settings import Settings, settings
from src.data_processing.dataset import get_dataloaders
from src.models.chess_model import FischerPolicyNet, get_available_device, load_model_weights


@dataclass(frozen=True)
class TestMetrics:
    """Aggregate metrics produced by evaluation on the held-out test split."""

    loss: float
    top1_accuracy: float
    top3_accuracy: float
    examples: int


def _move_batch(batch: tuple[Tensor, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    """Move a batch to the evaluation device."""
    boards, labels = batch
    non_blocking = device.type == "cuda"
    return (
        boards.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
    )


def _top_k_correct(logits: Tensor, labels: Tensor, k: int) -> int:
    """Return the number of labels contained in the logits' top-``k`` values."""
    effective_k = min(k, logits.size(1))
    top_indices = logits.topk(effective_k, dim=1).indices
    return top_indices.eq(labels.unsqueeze(1)).any(dim=1).sum().item()


def evaluate_policy(
    model: FischerPolicyNet,
    data_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    use_amp: bool,
) -> TestMetrics:
    """Evaluate a frozen policy on a DataLoader without updating its weights.

    Args:
        model: Loaded policy in evaluation mode.
        data_loader: Held-out DataLoader yielding states and action labels.
        criterion: Cross-entropy objective for raw policy logits.
        device: Compute device used for inference.
        use_amp: Whether to use supported GPU mixed-precision inference.

    Returns:
        Mean loss plus Top-1 and Top-3 action accuracy.

    Raises:
        ValueError: If ``data_loader`` has no examples.
    """
    model.eval()
    total_loss = 0.0
    top1_correct = 0
    top3_correct = 0
    example_count = 0

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="test", leave=False):
            boards, labels = _move_batch(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(boards)
                loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            top1_correct += _top_k_correct(logits, labels, k=1)
            top3_correct += _top_k_correct(logits, labels, k=3)
            example_count += batch_size

    if example_count == 0:
        raise ValueError("Test DataLoader is empty.")
    return TestMetrics(
        loss=total_loss / example_count,
        top1_accuracy=top1_correct / example_count,
        top3_accuracy=top3_correct / example_count,
        examples=example_count,
    )


def evaluate_bc(
    config: Settings = settings,
    checkpoint_path: Optional[Path] = None,
) -> TestMetrics:
    """Load the best BC checkpoint and evaluate it once on the test split.

    Args:
        config: Centralized paths and DataLoader configuration.
        checkpoint_path: Optional checkpoint override. Defaults to
            ``config.model_checkpoint_path``.

    Returns:
        Test-set loss, Top-1 accuracy, Top-3 accuracy, and sample count.
    """
    device = get_available_device()
    selected_checkpoint = checkpoint_path or (
        config.training_checkpoint_dir / "best_fischer_bc.pth"
    )
    data_loaders = get_dataloaders(
        file_path=config.training_data_path,
        batch_size=config.training_batch_size,
        num_workers=config.training_num_workers,
        seed=config.training_seed,
    )
    test_loader = data_loaders["test"]
    if len(test_loader) == 0:
        raise ValueError("The held-out test split has no batches to evaluate.")

    model = load_model_weights(selected_checkpoint, device)
    use_amp = config.training_use_amp and device.type in {"cuda", "mps"}
    metrics = evaluate_policy(
        model,
        test_loader,
        nn.CrossEntropyLoss(),
        device,
        use_amp,
    )
    print(f"Checkpoint: {selected_checkpoint}")
    print(f"Device: {device}")
    print(f"Test examples: {metrics.examples}")
    print(f"Test Loss: {metrics.loss:.4f}")
    print(f"Test Top-1: {metrics.top1_accuracy:.2%}")
    print(f"Test Top-3: {metrics.top3_accuracy:.2%}")
    return metrics


if __name__ == "__main__":
    evaluate_bc()
