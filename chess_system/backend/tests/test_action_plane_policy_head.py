"""Tests for the spatial 73-plane Fischer policy head."""

from __future__ import annotations

import torch

from src.data_processing.encoder import ACTION_SPACE_SIZE, NUM_MOVE_PLANES
from src.models.chess_model import FischerPolicyNet, _action_planes_to_logits


def test_action_plane_flattening_matches_move_label_order() -> None:
    """Keep rank, file, and move-plane order aligned with action labels."""
    action_planes = torch.zeros((1, NUM_MOVE_PLANES, 8, 8))
    rank, file, move_plane = 1, 4, 17  # e2 and an arbitrary action plane.
    expected_index = (rank * 8 + file) * NUM_MOVE_PLANES + move_plane
    action_planes[0, move_plane, rank, file] = 10.0

    logits = _action_planes_to_logits(action_planes)

    assert logits.shape == (1, ACTION_SPACE_SIZE)
    assert logits.argmax(dim=1).item() == expected_index


def test_action_plane_policy_emits_expected_logits_shape() -> None:
    """Return raw 4,672-class logits without the old dense output layer."""
    model = FischerPolicyNet(channels=32, residual_blocks=3, policy_channels=16)

    logits = model(torch.zeros((2, 18, 8, 8)))

    assert model.policy_head_type == "action_plane"
    assert hasattr(model, "policy_action_conv")
    assert not hasattr(model, "policy_linear")
    assert logits.shape == (2, ACTION_SPACE_SIZE)
