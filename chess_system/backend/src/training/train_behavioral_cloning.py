"""Behavioral Cloning training loop for the Fischer chess policy network.

Run from the ``backend`` directory with ``python -m
src.training.train_behavioral_cloning``. Training values are sourced from
``src.config.settings`` and can be overridden by environment variables.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config.settings import Settings, settings
from src.data_processing.dataset import get_dataloaders
from src.models.chess_model import FischerPolicyNet, get_available_device, set_random_seeds


@dataclass
class EpochMetrics:
    """Aggregate loss and accuracy produced by a train or validation epoch."""

    loss: float
    accuracy: float
    examples: int


@dataclass
class TrainingResult:
    """Summary returned after Behavioral Cloning training completes."""

    best_validation_loss: float
    final_epoch: int
    best_checkpoint_path: Path
    last_checkpoint_path: Path


def build_policy_model(config: Settings) -> FischerPolicyNet:
    """Create the policy architecture described by centralized settings.

    Args:
        config: Application settings containing model architecture fields.

    Returns:
        A newly initialized policy network.
    """
    return FischerPolicyNet(
        input_channels=config.model_input_channels,
        num_actions=config.model_num_actions,
        channels=config.model_channels,
        residual_blocks=config.model_residual_blocks,
        policy_channels=config.model_policy_channels,
    )


def _move_batch(batch: Tuple[Tensor, Tensor], device: torch.device) -> Tuple[Tensor, Tensor]:
    """Move one encoded-state batch to the requested device."""
    boards, labels = batch
    non_blocking = device.type == "cuda"
    return (
        boards.to(device, non_blocking=non_blocking),
        labels.to(device, non_blocking=non_blocking),
    )


def _run_epoch(
    model: FischerPolicyNet,
    data_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    optimizer: Optional[Optimizer] = None,
    amp_enabled: bool = False,
) -> EpochMetrics:
    """Run one train or evaluation epoch over a policy DataLoader.

    Args:
        model: Policy to optimize or evaluate.
        data_loader: Loader that yields ``(board_tensor, action_label)``.
        criterion: Classification objective for raw policy logits.
        device: Active compute device.
        optimizer: Optimizer. Omit it to run validation without gradients.
        amp_enabled: Whether CUDA autocast should be used.

    Returns:
        Mean loss, top-1 action accuracy, and number of examples.
    """
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct_predictions = 0
    example_count = 0
    progress = tqdm(data_loader, desc="train" if is_training else "validation", leave=False)

    with torch.set_grad_enabled(is_training):
        for batch in progress:
            boards, labels = _move_batch(batch, device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(boards)
                loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.detach().item() * batch_size
            correct_predictions += (logits.detach().argmax(dim=1) == labels).sum().item()
            example_count += batch_size
            progress.set_postfix(loss=f"{total_loss / example_count:.4f}")

    if example_count == 0:
        raise ValueError("DataLoader is empty; cannot train or evaluate an epoch.")
    return EpochMetrics(
        loss=total_loss / example_count,
        accuracy=correct_predictions / example_count,
        examples=example_count,
    )


def _checkpoint_paths(config: Settings) -> Tuple[Path, Path]:
    """Return stable paths for the best and most-recent training checkpoints."""
    checkpoint_dir = config.training_checkpoint_dir
    return checkpoint_dir / "fischer_bc_best.pt", checkpoint_dir / "fischer_bc_last.pt"


def _save_checkpoint(
    path: Path,
    model: FischerPolicyNet,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    epoch: int,
    best_validation_loss: float,
) -> None:
    """Persist all state required to reproduce or resume training.

    Args:
        path: Checkpoint destination.
        model: Policy whose parameters are saved.
        optimizer: Optimizer whose internal state is saved.
        scheduler: Learning-rate scheduler whose state is saved.
        epoch: Completed zero-based epoch number.
        best_validation_loss: Lowest validation loss observed so far.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.model_config(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
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
    """Restore model, optimizer, scheduler, and progress from a checkpoint.

    Args:
        checkpoint_path: Complete checkpoint created by :func:`_save_checkpoint`.
        model: Existing architecture to restore into.
        optimizer: Optimizer to restore.
        scheduler: Scheduler to restore.
        device: Device used for checkpoint mapping.

    Returns:
        The next epoch index and lowest validation loss stored in the checkpoint.

    Raises:
        FileNotFoundError: If the requested resume checkpoint is absent.
        ValueError: If the checkpoint does not include required training state.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint: Mapping[str, Any] = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "best_validation_loss",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(
            f"Resume checkpoint is incomplete; missing keys: {sorted(missing_keys)}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint["best_validation_loss"])


def _append_metrics(
    metrics_path: Path,
    epoch: int,
    train_metrics: EpochMetrics,
    validation_metrics: EpochMetrics,
    learning_rate: float,
) -> None:
    """Append one epoch's metrics to a CSV file, creating its header if needed."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "val_loss": validation_metrics.loss,
                "val_accuracy": validation_metrics.accuracy,
                "learning_rate": learning_rate,
            }
        )


def train_behavioral_cloning(config: Settings = settings) -> TrainingResult:
    """Train the Fischer policy by imitating moves from prepared DataLoaders.

    Args:
        config: Centralized settings for paths, architecture, and training.

    Returns:
        Final training summary, including best and latest checkpoint paths.
    """
    set_random_seeds(config.training_seed, config.training_deterministic)
    device = get_available_device()
    data_loaders = get_dataloaders(
        file_path=config.training_data_path,
        batch_size=config.training_batch_size,
        num_workers=config.training_num_workers,
        seed=config.training_seed,
    )
    if not data_loaders["train"] or not data_loaders["val"]:
        raise ValueError("Training and validation DataLoaders must both contain examples.")

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
    start_epoch = 0
    best_validation_loss = float("inf")
    if config.training_resume_path is not None:
        start_epoch, best_validation_loss = _load_resume_checkpoint(
            config.training_resume_path,
            model,
            optimizer,
            scheduler,
            device,
        )

    best_checkpoint_path, last_checkpoint_path = _checkpoint_paths(config)
    amp_enabled = config.training_use_amp and device.type == "cuda"
    for epoch in range(start_epoch, config.training_epochs):
        train_metrics = _run_epoch(
            model,
            data_loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            amp_enabled=amp_enabled,
        )
        validation_metrics = _run_epoch(model, data_loaders["val"], criterion, device)
        scheduler.step(validation_metrics.loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        _append_metrics(
            config.training_metrics_path,
            epoch,
            train_metrics,
            validation_metrics,
            learning_rate,
        )

        if validation_metrics.loss < best_validation_loss:
            best_validation_loss = validation_metrics.loss
            _save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_validation_loss,
            )
        _save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_validation_loss,
        )
        print(
            f"Epoch {epoch + 1}/{config.training_epochs} | "
            f"train_loss={train_metrics.loss:.4f} | val_loss={validation_metrics.loss:.4f} | "
            f"val_acc={validation_metrics.accuracy:.4f} | lr={learning_rate:.2e}"
        )

    return TrainingResult(
        best_validation_loss=best_validation_loss,
        final_epoch=config.training_epochs,
        best_checkpoint_path=best_checkpoint_path,
        last_checkpoint_path=last_checkpoint_path,
    )


if __name__ == "__main__":
    train_behavioral_cloning()
