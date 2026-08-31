"""Behavioral Cloning trainer for the Fischer chess policy.

Run from the ``backend`` directory with ``python -m src.training.train_bc``.
All paths, architecture values, and training hyperparameters are loaded from
``src.config.settings``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config.settings import Settings, settings
from src.data_processing.dataset import get_dataloaders
from src.models.chess_model import (
    FischerPolicyNet,
    LegalMoveCrossEntropyLoss,
    get_available_device,
    mask_illegal_logits,
    set_random_seeds,
)

_SCHEDULER_MONITOR = "val_top1_accuracy"


@dataclass(frozen=True)
class EpochMetrics:
    """Loss and ranking accuracy values accumulated over one epoch."""

    loss: float
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
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
        policy_dropout=config.model_policy_dropout,
        policy_head_type=config.model_policy_head_type,
    )


def _move_batch(
    batch: tuple[Tensor, Tensor, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Move board states, labels, and legal masks to ``device``."""
    boards, labels, legal_move_masks = batch
    non_blocking = device.type == "cuda"
    return (
        boards.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
        legal_move_masks.to(device, non_blocking=non_blocking),
    )


def top_k_accuracy(
    logits: Tensor,
    labels: Tensor,
    k: int,
    legal_move_mask: Optional[Tensor] = None,
) -> float:
    """Calculate Top-k accuracy, optionally ranking only legal chess moves.

    Args:
        logits: Unnormalized policy outputs with shape ``[batch_size, actions]``.
        labels: Target action indices with shape ``[batch_size]``.
        k: Number of highest-ranked actions to consider.
        legal_move_mask: Boolean legal-action tensor with the same shape as
            ``logits``. Illegal actions are excluded before ranking.

    Returns:
        Proportion of labels present in the Top-k predictions.

    Raises:
        ValueError: If tensor shapes are incompatible or ``k`` is invalid.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_actions].")
    if labels.ndim != 1 or labels.size(0) != logits.size(0):
        raise ValueError("labels must have shape [batch_size].")
    if k < 1:
        raise ValueError("k must be at least 1.")
    if legal_move_mask is not None:
        logits = mask_illegal_logits(logits, legal_move_mask)

    effective_k = min(k, logits.size(1))
    predicted_indices = logits.topk(effective_k, dim=1).indices
    correct = predicted_indices.eq(labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def _run_epoch(
    model: FischerPolicyNet,
    data_loader: DataLoader,
    criterion: LegalMoveCrossEntropyLoss,
    device: torch.device,
    optimizer: Optional[Optimizer],
    grad_scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> EpochMetrics:
    """Run one training or validation phase and collect ranking metrics.

    Args:
        model: Policy network to train or validate.
        data_loader: Loader yielding board tensors, action labels, and legal masks.
        criterion: Legal-move cross-entropy objective for raw model logits.
        device: Device on which inference and training are performed.
        optimizer: Optimizer for the train phase; ``None`` for validation.
        grad_scaler: CUDA gradient scaler used by mixed-precision training.
        use_amp: Whether the active device should run autocast operations.

    Returns:
        Mean cross-entropy loss and Top-1/Top-3/Top-5 action accuracies.
    """
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    top1_correct = 0.0
    top3_correct = 0.0
    top5_correct = 0.0
    example_count = 0
    phase_name = "train" if is_training else "validation"
    progress = tqdm(data_loader, desc=phase_name, leave=False)

    with torch.set_grad_enabled(is_training):
        for batch in progress:
            boards, labels, legal_move_masks = _move_batch(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                raw_logits = model(boards)
                loss = criterion(raw_logits, labels, legal_move_masks)

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
            detached_logits = raw_logits.detach()
            top1_correct += batch_size * top_k_accuracy(
                detached_logits, labels, k=1, legal_move_mask=legal_move_masks
            )
            top3_correct += batch_size * top_k_accuracy(
                detached_logits, labels, k=3, legal_move_mask=legal_move_masks
            )
            top5_correct += batch_size * top_k_accuracy(
                detached_logits, labels, k=5, legal_move_mask=legal_move_masks
            )
            example_count += batch_size
            progress.set_postfix(
                loss=f"{total_loss / example_count:.4f}",
                top1=f"{top1_correct / example_count:.3f}",
                top3=f"{top3_correct / example_count:.3f}",
                top5=f"{top5_correct / example_count:.3f}",
            )

    if example_count == 0:
        raise ValueError(f"{phase_name} DataLoader is empty.")
    return EpochMetrics(
        loss=total_loss / example_count,
        top1_accuracy=top1_correct / example_count,
        top3_accuracy=top3_correct / example_count,
        top5_accuracy=top5_correct / example_count,
        examples=example_count,
    )


def _checkpoint_paths(config: Settings) -> tuple[Path, Path]:
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
    epochs_without_improvement: int,
) -> None:
    """Save all model and optimization state needed to resume training.

    Args:
        path: Destination checkpoint file.
        model: Current policy model.
        optimizer: Optimizer with current momentum state.
        scheduler: Learning-rate scheduler state.
        epoch: Completed zero-based epoch index.
        best_validation_top1_accuracy: Highest validation Top-1 accuracy so far.
        epochs_without_improvement: Consecutive epochs without a higher Top-1 score.
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
            "epochs_without_improvement": epochs_without_improvement,
            "scheduler_monitor": _SCHEDULER_MONITOR,
        },
        path,
    )


def _load_resume_checkpoint(
    checkpoint_path: Path,
    model: FischerPolicyNet,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
) -> tuple[int, float, int]:
    """Restore a complete checkpoint and return early-stopping state.

    Args:
        checkpoint_path: Complete checkpoint generated by this module.
        model: Model instance to receive saved weights.
        optimizer: Optimizer instance to restore.
        scheduler: Scheduler instance to restore.
        device: Device used to map checkpoint tensors.

    Returns:
        The next epoch index, best validation Top-1 accuracy, and stale-epoch
        count.

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
        "epochs_without_improvement",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(f"Resume checkpoint is incomplete; missing: {sorted(missing_keys)}")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # Older checkpoints scheduled on validation loss (``mode='min'``). Their
    # plateau state must not be restored after switching to Top-1 accuracy,
    # otherwise the scheduler would compare incompatible values on resume.
    if checkpoint.get("scheduler_monitor") == _SCHEDULER_MONITOR:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_validation_top1_accuracy"]),
        int(checkpoint["epochs_without_improvement"]),
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
                "train_top5_accuracy",
                "val_loss",
                "val_top1_accuracy",
                "val_top3_accuracy",
                "val_top5_accuracy",
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
                "train_top5_accuracy": train_metrics.top5_accuracy,
                "val_loss": validation_metrics.loss,
                "val_top1_accuracy": validation_metrics.top1_accuracy,
                "val_top3_accuracy": validation_metrics.top3_accuracy,
                "val_top5_accuracy": validation_metrics.top5_accuracy,
                "learning_rate": learning_rate,
            }
        )


def train_bc(config: Settings = settings) -> TrainingResult:
    """Train a Fischer policy with Behavioral Cloning.

    The best checkpoint is written as ``best_fischer_bc.pth`` whenever the
    validation Top-1 accuracy improves. Top-1, Top-3, and Top-5 accuracy are
    logged as style-learning diagnostics. Training stops after the configured
    number of consecutive epochs without a higher validation Top-1 score.

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
        strict_fen_disjoint=config.training_strict_fen_disjoint,
    )
    if len(data_loaders["train"]) == 0 or len(data_loaders["val"]) == 0:
        raise ValueError("Both train and validation DataLoaders must contain examples.")

    model = build_policy_model(config).to(device)
    if not 0.0 <= config.training_label_smoothing <= 1.0:
        raise ValueError("training_label_smoothing must be in the range [0.0, 1.0].")
    if config.training_early_stopping_patience < 1:
        raise ValueError("training_early_stopping_patience must be at least 1.")
    criterion = LegalMoveCrossEntropyLoss(
        label_smoothing=config.training_label_smoothing
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.training_learning_rate,
        weight_decay=config.training_weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
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
    epochs_without_improvement = 0
    if config.training_resume_path is not None:
        (
            start_epoch,
            best_validation_top1_accuracy,
            epochs_without_improvement,
        ) = _load_resume_checkpoint(
            config.training_resume_path,
            model,
            optimizer,
            scheduler,
            device,
        )

    best_checkpoint_path, last_checkpoint_path = _checkpoint_paths(config)
    final_epoch = start_epoch
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
        scheduler.step(validation_metrics.top1_accuracy)
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
            epochs_without_improvement = 0
            _save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_validation_top1_accuracy,
                epochs_without_improvement,
            )
        else:
            epochs_without_improvement += 1
        _save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_validation_top1_accuracy,
            epochs_without_improvement,
        )
        print(
            f"Epoch {epoch + 1}/{config.training_epochs} | "
            f"Train Loss: {train_metrics.loss:.4f} | "
            f"Train Top-1: {train_metrics.top1_accuracy:.2%} | "
            f"Train Top-3: {train_metrics.top3_accuracy:.2%} | "
            f"Train Top-5: {train_metrics.top5_accuracy:.2%} | "
            f"Val Loss: {validation_metrics.loss:.4f} | "
            f"Val Top-1: {validation_metrics.top1_accuracy:.2%} | "
            f"Val Top-3: {validation_metrics.top3_accuracy:.2%} | "
            f"Val Top-5: {validation_metrics.top5_accuracy:.2%} | "
            f"LR: {learning_rate:.2e}"
        )
        final_epoch = epoch + 1
        if epochs_without_improvement >= config.training_early_stopping_patience:
            print(
                "Early stopping: validation Top-1 accuracy did not improve for "
                f"{epochs_without_improvement} epoch(s)."
            )
            break

    return TrainingResult(
        best_validation_top1_accuracy=best_validation_top1_accuracy,
        final_epoch=final_epoch,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
    )


if __name__ == "__main__":
    train_bc()
