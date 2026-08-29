"""Tests for legal-action cross-entropy with label smoothing."""

import pytest
import torch

from src.models.chess_model import LegalMoveCrossEntropyLoss


def test_label_smoothing_ignores_negative_infinity_illegal_logits() -> None:
    """A finite loss is produced when smoothing is restricted to legal moves."""
    logits = torch.tensor([[0.2, 8.0, 1.3], [2.0, 3.0, -4.0]])
    labels = torch.tensor([2, 0], dtype=torch.long)
    legal_move_mask = torch.tensor([[True, False, True], [True, True, False]])

    loss = LegalMoveCrossEntropyLoss(label_smoothing=0.1)(
        logits,
        labels,
        legal_move_mask,
    )

    assert torch.isfinite(loss)


def test_illegal_target_raises_clear_error() -> None:
    """Dataset/action mapping errors must not silently yield an infinite loss."""
    logits = torch.tensor([[0.2, 8.0, 1.3]])
    labels = torch.tensor([1], dtype=torch.long)
    legal_move_mask = torch.tensor([[True, False, True]])

    with pytest.raises(ValueError, match="target action must be legal"):
        LegalMoveCrossEntropyLoss()(logits, labels, legal_move_mask)
