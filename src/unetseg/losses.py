"""Losses for binary segmentation, all on logits of shape B x 1 x H x W.

BCE treats every pixel as an independent coin flip:
    L_bce = -mean[ y log p + (1-y) log(1-p) ],  p = sigmoid(z)
When the object is small, the background pixels dominate that mean and a model that
predicts "all background" already has a low loss.

Soft Dice measures overlap of the whole mask, so it does not care how much background
there is:
    Dice = (2 sum(p y) + eps) / (sum(p) + sum(y) + eps),   L_dice = 1 - Dice
It is computed per image and averaged. Dice alone has noisy gradients early in
training (when p is ~0.5 everywhere), which is why BCE + Dice is the usual default:
BCE gives smooth per-pixel gradients, Dice pushes the overlap directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits).flatten(1)
    y = target.flatten(1)
    inter = (p * y).sum(1)
    dice = (2 * inter + eps) / (p.sum(1) + y.sum(1) + eps)
    return 1 - dice.mean()


class SegLoss(nn.Module):
    """`kind` is "bce", "dice" or "bce+dice" (equal weights)."""

    def __init__(self, kind: str = "bce+dice"):
        super().__init__()
        if kind not in {"bce", "dice", "bce+dice"}:
            raise ValueError(f"loss must be bce, dice or bce+dice, got {kind!r}")
        self.kind = kind

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = logits.new_zeros(())
        if "bce" in self.kind:
            loss = loss + F.binary_cross_entropy_with_logits(logits, target)
        if "dice" in self.kind:
            loss = loss + soft_dice_loss(logits, target)
        return loss
