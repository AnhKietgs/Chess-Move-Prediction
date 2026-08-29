"""Unit tests for Behavioral Cloning ranking metrics."""

import torch

from src.training.train_bc import top_k_accuracy


def test_top_k_accuracy_excludes_illegal_actions() -> None:
    """An illegal high-logit action must not affect the Top-k ranking."""
    logits = torch.tensor([[0.0, 100.0, 2.0]])
    labels = torch.tensor([2])
    legal_move_mask = torch.tensor([[True, False, True]])

    assert top_k_accuracy(logits, labels, k=1, legal_move_mask=legal_move_mask) == 1.0


def test_top_k_accuracy_without_mask_uses_all_actions() -> None:
    """The optional mask leaves ordinary classification behavior available."""
    logits = torch.tensor([[0.0, 4.0, 2.0], [5.0, 1.0, 0.0]])
    labels = torch.tensor([2, 0])

    assert top_k_accuracy(logits, labels, k=1) == 0.5
    assert top_k_accuracy(logits, labels, k=2) == 1.0
