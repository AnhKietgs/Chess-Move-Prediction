"""Behavioral Cloning trainer for the Fischer chess policy.

Run from the ``backend`` directory with ``python -m src.training.train_bc``.
All paths, architecture values, and training hyperparameters are loaded from
``src.config.settings``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config.settings import Settings, settings
from src.data_processing.dataset import get_dataloaders
from src.models.chess_model import FischerPolicyNet, get_available_device, set_random_seeds


@dataclass(frozen=True)
class EpochMetrics:
    """Loss and ranking accuracy values accumulated over one epoch."""

    loss: float
    top1_accuracy: float
    top3_accuracy: float
    examples: int


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed Behavioral Cloning training run."""

    best_validation_top1_accuracy: float
    final_epoch: int
    best_checkpoint_path: Path
    last_checkpoint_path: Path


def build_policy_model(config: Settings) -> FischerPolicyNet:
    """Create a Fischer policy with architecture values from configuration.

    Args:
        config: Centralized application and training configuration.

    Returns:
        An untrained Fischer policy network.
    """
    return FischerPolicyNet(
        input_channels=config.model_input_channels,
        num_actions=config.model_num_actions,
        channels=config.model_channels,
        residual_blocks=config.model_residual_blocks,
        policy_channels=config.model_policy_channels,
    )


def _move_batch(batch: Tuple[Tensor, Tensor], device: torch.device) -> Tuple[Tensor, Tensor]:
    """Move an input-and-label batch to ``device``."""
    boards, labels = batch
    non_blocking = device.type == "cuda"
    return (
        boards.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
    )


def _top_k_correct(logits: Tensor, labels: Tensor, k: int) -> int:
    """Count examples whose label occurs in the model's top-``k`` logits."""
    effective_k = min(k, logits.size(1))
    predicted_indices = logits.topk(effective_k, dim=1).indices
    return predicted_indices.eq(labels.unsqueeze(1)).any(dim=1).sum().item()


def _run_epoch(
    model: FischerPolicyNet,
    data_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    optimizer: Optional[Optimizer],
    grad_scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> EpochMetrics:
    """Run one training or validation phase and collect ranking metrics.

    Args:
        model: Policy network to train or validate.
        data_loader: Loader yielding board tensors and integer action labels.
        criterion: Cross-entropy objective for raw model logits.
        device: Device on which inference and training are performed.
        optimizer: Optimizer for the train phase; ``None`` for validation.
        grad_scaler: CUDA gradient scaler used by mixed-precision training.
        use_amp: Whether the active device should run autocast operations.

    Returns:
        Mean cross-entropy loss and Top-1/Top-3 action accuracies.
    """
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    top1_correct = 0
    top3_correct = 0
    example_count = 0
    phase_name = "train" if is_training else "validation"
    progress = tqdm(data_loader, desc=phase_name, leave=False)

    with torch.set_grad_enabled(is_training):
        for batch in progress:
            boards, labels = _move_batch(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(boards)
                loss = criterion(logits, labels)

            if optimizer is not None:
                if grad_scaler.is_enabled():
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.detach().item() * batch_size
            top1_correct += _top_k_correct(logits.detach(), labels, k=1)
            top3_correct += _top_k_correct(logits.detach(), labels, k=3)
            example_count += batch_size
            progress.set_postfix(
                loss=f"{total_loss / example_count:.4f}",
                top1=f"{top1_correct / example_count:.3f}",
                top3=f"{top3_correct / example_count:.3f}",
            )

    if example_count == 0:
        raise ValueError(f"{phase_name} DataLoader is empty.")
    return EpochMetrics(
        loss=total_loss / example_count,
        top1_accuracy=top1_correct / example_count,
        top3_accuracy=top3_correct / example_count,
        examples=example_count,
    )


def _checkpoint_paths(config: Settings) -> Tuple[Path, Path]:
    """Return checkpoint paths for the best and most-recent policy states."""
    return (
        config.training_checkpoint_dir / "best_fischer_bc.pth",
        config.training_checkpoint_dir / "last_fischer_bc.pth",
    )


def _save_checkpoint(
    path: Path,
    model: FischerPolicyNet,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_validation_top1_accuracy: float,
) -> None:
    """Save all model and optimization state needed to resume training.

    Args:
        path: Destination checkpoint file.
        model: Current policy model.
        optimizer: Optimizer with current momentum state.
        scheduler: Learning-rate scheduler state.
        epoch: Completed zero-based epoch index.
        best_validation_top1_accuracy: Best validation Top-1 score so far.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_validation_top1_accuracy": best_validation_top1_accuracy,
        },
        path,
    )


def _load_resume_checkpoint(
    checkpoint_path: Path,
    model: FischerPolicyNet,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
) -> Tuple[int, float]:
    """Restore a complete checkpoint and return next epoch and best Top-1.

    Args:
        checkpoint_path: Complete checkpoint generated by this module.
        model: Model instance to receive saved weights.
        optimizer: Optimizer instance to restore.
        scheduler: Scheduler instance to restore.
        device: Device used to map checkpoint tensors.

    Returns:
        The next epoch index and the saved best validation Top-1 accuracy.

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
        ValueError: If the file lacks complete resume state.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    try:
        loaded_checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        loaded_checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(loaded_checkpoint, Mapping):
        raise ValueError("Resume checkpoint must be a mapping.")
    checkpoint: Mapping[str, Any] = loaded_checkpoint
    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "best_validation_top1_accuracy",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(f"Resume checkpoint is incomplete; missing: {sorted(missing_keys)}")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_validation_top1_accuracy"]),
    )


def _append_metrics(
    metrics_path: Path,
    epoch: int,
    train_metrics: EpochMetrics,
    validation_metrics: EpochMetrics,
    learning_rate: float,
) -> None:
    """Append epoch metrics to CSV, creating the file and header when absent."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    create_header = not metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_top1_accuracy",
                "train_top3_accuracy",
                "val_loss",
                "val_top1_accuracy",
                "val_top3_accuracy",
                "learning_rate",
            ],
        )
        if create_header:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics.loss,
                "train_top1_accuracy": train_metrics.top1_accuracy,
                "train_top3_accuracy": train_metrics.top3_accuracy,
                "val_loss": validation_metrics.loss,
                "val_top1_accuracy": validation_metrics.top1_accuracy,
                "val_top3_accuracy": validation_metrics.top3_accuracy,
                "learning_rate": learning_rate,
            }
        )


def train_bc(config: Settings = settings) -> TrainingResult:
    """Train a Fischer policy with Behavioral Cloning.

    The best checkpoint is written as ``best_fischer_bc.pth`` whenever the
    validation Top-1 accuracy improves. Top-3 accuracy is calculated and
    logged each epoch as a style-learning diagnostic.

    Args:
        config: Centralized settings for architecture and training values.

    Returns:
        Summary containing the best score and checkpoint locations.
    """
    set_random_seeds(config.training_seed, config.training_deterministic)
    device = get_available_device()
    data_loaders = get_dataloaders(
        file_path=config.training_data_path,
        batch_size=config.training_batch_size,
        num_workers=config.training_num_workers,
        seed=config.training_seed,
    )
    if len(data_loaders["train"]) == 0 or len(data_loaders["val"]) == 0:
        raise ValueError("Both train and validation DataLoaders must contain examples.")

    model = build_policy_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.training_learning_rate,
        weight_decay=config.training_weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training_scheduler_factor,
        patience=config.training_scheduler_patience,
        min_lr=config.training_min_learning_rate,
    )

    use_amp = config.training_use_amp and device.type in {"cuda", "mps"}
    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and device.type == "cuda",
    )
    start_epoch = 0
    best_validation_top1_accuracy = float("-inf")
    if config.training_resume_path is not None:
        start_epoch, best_validation_top1_accuracy = _load_resume_checkpoint(
            config.training_resume_path,
            model,
            optimizer,
            scheduler,
            device,
        )

    best_checkpoint_path, last_checkpoint_path = _checkpoint_paths(config)
    for epoch in range(start_epoch, config.training_epochs):
        train_metrics = _run_epoch(
            model,
            data_loaders["train"],
            criterion,
            device,
            optimizer,
            grad_scaler,
            use_amp,
        )
        validation_metrics = _run_epoch(
            model,
            data_loaders["val"],
            criterion,
            device,
            optimizer=None,
            grad_scaler=grad_scaler,
            use_amp=use_amp,
        )
        scheduler.step(validation_metrics.loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        _append_metrics(
            config.training_metrics_path,
            epoch,
            train_metrics,
            validation_metrics,
            learning_rate,
        )

        if validation_metrics.top1_accuracy > best_validation_top1_accuracy:
            best_validation_top1_accuracy = validation_metrics.top1_accuracy
            _save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_validation_top1_accuracy,
            )
        _save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_validation_top1_accuracy,
        )
        print(
            f"Epoch {epoch + 1}/{config.training_epochs} | "
            f"Train Loss: {train_metrics.loss:.4f} | "
            f"Train Top-1: {train_metrics.top1_accuracy:.2%} | "
            f"Train Top-3: {train_metrics.top3_accuracy:.2%} | "
            f"Val Loss: {validation_metrics.loss:.4f} | "
            f"Val Top-1: {validation_metrics.top1_accuracy:.2%} | "
            f"Val Top-3: {validation_metrics.top3_accuracy:.2%} | "
            f"LR: {learning_rate:.2e}"
        )

    return TrainingResult(
        best_validation_top1_accuracy=best_validation_top1_accuracy,
        final_epoch=config.training_epochs,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
    )


if __name__ == "__main__":
    train_bc()
