"""Neural-network policy architecture for Fischer-style chess play."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from src.data_processing.encoder import ACTION_SPACE_SIZE, NUM_CHANNELS, NUM_MOVE_PLANES


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


def mask_illegal_logits(logits: Tensor, legal_move_mask: Tensor) -> Tensor:
    """Set logits for illegal actions to negative infinity.

    Args:
        logits: Raw policy logits of shape ``[batch_size, num_actions]``.
        legal_move_mask: Boolean mask with exactly the same shape as
            ``logits``; ``True`` marks an action legal for that sample.

    Returns:
        Policy logits with all illegal actions replaced by ``-inf``. The
        returned logits can be passed directly to argmax or to cross-entropy
        without label smoothing.

    Raises:
        ValueError: If shapes differ or any sample has no legal action.
    """
    if logits.ndim != 2 or legal_move_mask.shape != logits.shape:
        raise ValueError(
            "logits and legal_move_mask must have matching shape "
            "[batch_size, num_actions]."
        )
    if not torch.all(legal_move_mask.any(dim=1)):
        raise ValueError("Every sample must include at least one legal action.")
    return logits.masked_fill(~legal_move_mask.to(dtype=torch.bool), float("-inf"))


class LegalMoveCrossEntropyLoss(nn.Module):
    """Cross-entropy that applies label smoothing only across legal actions.

    PyTorch's built-in label smoothing distributes probability over every
    action class. That is incompatible with ``-inf`` illegal-action logits.
    This criterion instead distributes its smoothing mass uniformly over the
    legal moves for each individual chess position.

    Args:
        label_smoothing: Probability mass distributed uniformly among legal
            actions. Must be in the range ``[0, 1]``.
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError("label_smoothing must be in the range [0.0, 1.0].")
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        legal_move_mask: Tensor,
    ) -> Tensor:
        """Return mean legal-action cross-entropy for a policy batch.

        Args:
            logits: Raw policy logits with shape ``[batch_size, num_actions]``.
            labels: Legal target action indices with shape ``[batch_size]``.
            legal_move_mask: Boolean legal-action mask matching ``logits``.

        Returns:
            A finite scalar loss when every target action is legal.

        Raises:
            ValueError: If labels are malformed, out of range, or illegal.
        """
        masked_logits = mask_illegal_logits(logits, legal_move_mask)
        if labels.ndim != 1 or labels.size(0) != logits.size(0):
            raise ValueError("labels must have shape [batch_size].")
        if labels.dtype != torch.long:
            raise ValueError("labels must have dtype torch.long.")
        if torch.any(labels < 0) or torch.any(labels >= logits.size(1)):
            raise ValueError("labels contain an action index outside the action space.")

        legal_mask = legal_move_mask.to(device=logits.device, dtype=torch.bool)
        target_is_legal = legal_mask.gather(1, labels.unsqueeze(1)).squeeze(1)
        if not torch.all(target_is_legal):
            raise ValueError("Every target action must be legal for its board position.")

        log_probabilities = functional.log_softmax(masked_logits, dim=1)
        negative_log_likelihood = -log_probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        if self.label_smoothing == 0.0:
            return negative_log_likelihood.mean()

        legal_counts = legal_mask.sum(dim=1)
        legal_log_probability_mean = (
            log_probabilities.masked_fill(~legal_mask, 0.0).sum(dim=1) / legal_counts
        )
        smoothed_loss = (
            (1.0 - self.label_smoothing) * negative_log_likelihood
            - self.label_smoothing * legal_log_probability_mean
        )
        return smoothed_loss.mean()


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


def _action_planes_to_logits(action_planes: Tensor) -> Tensor:
    """Flatten action planes using the project's exact move-label order.

    The action encoder defines ``index = from_square * 73 + move_plane``.
    A python-chess square index is ``rank * 8 + file``. Moving the plane
    channel last before flattening therefore preserves that mapping exactly.

    Args:
        action_planes: Tensor with shape ``[batch_size, 73, 8, 8]``.

    Returns:
        Raw policy logits with shape ``[batch_size, 4672]``.

    Raises:
        ValueError: If the supplied tensor does not have 73 action planes.
    """
    if action_planes.ndim != 4 or action_planes.shape[1:] != (NUM_MOVE_PLANES, 8, 8):
        raise ValueError(
            "action_planes must have shape "
            f"[batch_size, {NUM_MOVE_PLANES}, 8, 8]."
        )
    return action_planes.permute(0, 2, 3, 1).contiguous().view(
        action_planes.size(0), ACTION_SPACE_SIZE
    )


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
        policy_dropout: Dropout probability before policy-logit generation.
        policy_head_type: ``"action_plane"`` emits 73 move planes for each
            source square. ``"dense_legacy"`` exists only to load old
            dense-head checkpoints.
    """

    def __init__(
        self,
        input_channels: int = NUM_CHANNELS,
        num_actions: int = ACTION_SPACE_SIZE,
        channels: int = 64,
        residual_blocks: int = 4,
        policy_channels: int = 32,
        policy_dropout: float = 0.4,
        policy_head_type: Literal["action_plane", "dense_legacy"] = "action_plane",
    ) -> None:
        super().__init__()
        if min(input_channels, num_actions, channels, residual_blocks, policy_channels) <= 0:
            raise ValueError("All FischerPolicyNet dimensions must be positive.")
        if not 0.0 <= policy_dropout < 1.0:
            raise ValueError("policy_dropout must be in the range [0.0, 1.0).")
        if policy_head_type not in {"action_plane", "dense_legacy"}:
            raise ValueError(
                "policy_head_type must be either 'action_plane' or 'dense_legacy'."
            )
        if policy_head_type == "action_plane" and num_actions != ACTION_SPACE_SIZE:
            raise ValueError(
                "action_plane policy heads require the 4672-action chess encoding."
            )

        self.input_channels = input_channels
        self.num_actions = num_actions
        self.channels = channels
        self.residual_blocks = residual_blocks
        self.policy_channels = policy_channels
        self.policy_dropout = policy_dropout
        self.policy_head_type = policy_head_type

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
        self.policy_dropout_layer = nn.Dropout(p=policy_dropout)
        if policy_head_type == "action_plane":
            self.policy_action_conv = nn.Conv2d(
                policy_channels,
                NUM_MOVE_PLANES,
                kernel_size=1,
            )
        else:
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
        policy = self.policy_dropout_layer(policy)
        if self.policy_head_type == "action_plane":
            # Shape after Conv2d: [batch_size, 73, 8, 8].
            action_planes = self.policy_action_conv(policy)
            return _action_planes_to_logits(action_planes)

        policy = torch.flatten(policy, start_dim=1)
        return self.policy_linear(policy)  # [batch_size, num_actions], raw logits

    def model_config(self) -> dict[str, int | float | str]:
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
            "policy_dropout": self.policy_dropout,
            "policy_head_type": self.policy_head_type,
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
    model_config: Mapping[str, int | float | str] = {}
    state_dict: Mapping[str, Tensor]
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        model_config = checkpoint.get("model_config", {})
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, Mapping):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported model checkpoint format: {checkpoint_path}")

    model_kwargs = dict(model_config)
    # Earlier checkpoints do not contain model_policy_head_type. Detect their
    # dense output layer so the established baseline remains deployable.
    if (
        "policy_head_type" not in model_kwargs
        and any(key.startswith("policy_linear.") for key in state_dict)
    ):
        model_kwargs["policy_head_type"] = "dense_legacy"

    try:
        model = FischerPolicyNet(**model_kwargs)
    except TypeError as exc:
        raise ValueError("Checkpoint contains invalid model_config metadata.") from exc
    model.load_state_dict(state_dict)
    return model.to(target_device).eval()
