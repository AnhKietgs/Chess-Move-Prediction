"""Neural-network policy architecture for Fischer-style chess play."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn

from src.data_processing.encoder import ACTION_SPACE_SIZE, NUM_CHANNELS


def get_available_device() -> torch.device:
    """Return the best available PyTorch device: CUDA, then MPS, then CPU.

    Returns:
        The selected compute device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_random_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch random generators.

    Args:
        seed: Non-negative seed used by all supported random generators.
        deterministic: Whether to prefer deterministic PyTorch algorithms.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


class _ResidualBlock(nn.Module):
    """Two-convolution residual block preserving the board's spatial shape."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the residual transformation.

        Args:
            inputs: Feature maps of shape ``[batch_size, channels, 8, 8]``.

        Returns:
            Feature maps with the same shape as ``inputs``.
        """
        residual = inputs
        # Shape after Conv2d: [batch_size, channels, 8, 8].
        outputs = self.conv1(inputs)
        outputs = self.activation(self.batch_norm1(outputs))
        # Shape after Conv2d: [batch_size, channels, 8, 8].
        outputs = self.conv2(outputs)
        outputs = self.batch_norm2(outputs)
        return self.activation(outputs + residual)


class FischerPolicyNet(nn.Module):
    """Residual CNN that maps an encoded board state to move-policy logits.

    The model intentionally emits unnormalized logits. This is the expected
    input for :class:`torch.nn.CrossEntropyLoss` and retains compatibility
    with future PPO policy distributions.

    Args:
        input_channels: Number of channels in the board encoding.
        num_actions: Size of the discrete move action space.
        channels: Width of the residual trunk.
        residual_blocks: Number of residual blocks in the trunk.
        policy_channels: Width of the 1x1 policy head convolution.
    """

    def __init__(
        self,
        input_channels: int = NUM_CHANNELS,
        num_actions: int = ACTION_SPACE_SIZE,
        channels: int = 128,
        residual_blocks: int = 8,
        policy_channels: int = 32,
    ) -> None:
        super().__init__()
        if min(input_channels, num_actions, channels, residual_blocks, policy_channels) <= 0:
            raise ValueError("All FischerPolicyNet dimensions must be positive.")

        self.input_channels = input_channels
        self.num_actions = num_actions
        self.channels = channels
        self.residual_blocks = residual_blocks
        self.policy_channels = policy_channels

        self.input_conv = nn.Conv2d(
            input_channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.input_batch_norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)
        self.residual_tower = nn.Sequential(
            *[_ResidualBlock(channels) for _ in range(residual_blocks)]
        )

        self.policy_conv = nn.Conv2d(channels, policy_channels, kernel_size=1, bias=False)
        self.policy_batch_norm = nn.BatchNorm2d(policy_channels)
        self.policy_linear = nn.Linear(policy_channels * 8 * 8, num_actions)

    def forward(self, board_tensor: Tensor) -> Tensor:
        """Return raw move logits for encoded board states.

        Args:
            board_tensor: Board tensor of shape
                ``[batch_size, input_channels, 8, 8]``.

        Returns:
            Raw logits of shape ``[batch_size, num_actions]``. No Softmax is
            applied, so the result can be supplied directly to
            :class:`torch.nn.CrossEntropyLoss`.

        Raises:
            ValueError: If the input is not a four-dimensional 8x8 board
                tensor with the configured channel count.
        """
        if board_tensor.ndim != 4:
            raise ValueError("board_tensor must have shape [batch_size, channels, 8, 8].")
        if board_tensor.shape[1] != self.input_channels or board_tensor.shape[2:] != (8, 8):
            raise ValueError(
                "Expected board_tensor shape [batch_size, "
                f"{self.input_channels}, 8, 8], got {tuple(board_tensor.shape)}."
            )

        # Shape after Conv2d: [batch_size, channels, 8, 8].
        features = self.input_conv(board_tensor)
        features = self.activation(self.input_batch_norm(features))
        features = self.residual_tower(features)
        # Shape after Conv2d: [batch_size, policy_channels, 8, 8].
        policy = self.policy_conv(features)
        policy = self.activation(self.policy_batch_norm(policy))
        policy = torch.flatten(policy, start_dim=1)  # [batch_size, policy_channels * 8 * 8]
        return self.policy_linear(policy)  # [batch_size, num_actions], raw logits

    def model_config(self) -> dict[str, int]:
        """Return architecture values needed to rebuild this policy.

        Returns:
            Serializable architecture configuration.
        """
        return {
            "input_channels": self.input_channels,
            "num_actions": self.num_actions,
            "channels": self.channels,
            "residual_blocks": self.residual_blocks,
            "policy_channels": self.policy_channels,
        }


def _read_checkpoint(model_path: Path, device: torch.device) -> Any:
    """Load a checkpoint while supporting currently maintained PyTorch releases."""
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def load_model_weights(
    model_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> FischerPolicyNet:
    """Build a policy model and load its weights from a saved checkpoint.

    Both complete training checkpoints and plain ``state_dict`` files are
    supported. A complete checkpoint restores the exact saved architecture;
    a plain state dict uses the default policy architecture.

    Args:
        model_path: Path to the checkpoint or model state dictionary.
        device: Target device. When omitted, CUDA, MPS, then CPU is selected.

    Returns:
        An evaluation-mode :class:`FischerPolicyNet` on ``device``.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        ValueError: If checkpoint architecture metadata is invalid.
    """
    checkpoint_path = Path(model_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")

    target_device = device or get_available_device()
    checkpoint = _read_checkpoint(checkpoint_path, target_device)
    model_config: Mapping[str, int] = {}
    state_dict: Mapping[str, Tensor]
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        model_config = checkpoint.get("model_config", {})
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, Mapping):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported model checkpoint format: {checkpoint_path}")

    try:
        model = FischerPolicyNet(**dict(model_config))
    except TypeError as exc:
        raise ValueError("Checkpoint contains invalid model_config metadata.") from exc
    model.load_state_dict(state_dict)
    return model.to(target_device).eval()
