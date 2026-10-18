"""Pixel metrics for binary masks, from one confusion matrix.

Everything comes from four counts summed over the whole evaluation set:

    TP: predicted 1, true 1     FP: predicted 1, true 0
    FN: predicted 0, true 1     TN: predicted 0, true 0

    IoU (Jaccard)   = TP / (TP + FP + FN)
    Dice (= F1)     = 2TP / (2TP + FP + FN)
    pixel accuracy  = (TP + TN) / all
    precision       = TP / (TP + FP)      "when I say object, am I right?"
    recall          = TP / (TP + FN)      "how much of the object did I find?"

Pixel accuracy is reported but not used to pick models: with 75% background a model
that predicts nothing scores 75%. IoU and Dice ignore TN and are the numbers to read.
`mean_iou_per_image` averages the IoU of each image instead (small objects count as
much as big ones).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.bool()
        target = target.bool()
        self.tp += int((pred & target).sum())
        self.fp += int((pred & ~target).sum())
        self.fn += int((~pred & target).sum())
        self.tn += int((~pred & ~target).sum())

    def matrix(self) -> np.ndarray:
        """Rows = true class (0, 1), columns = predicted class (0, 1)."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]], dtype=np.int64)

    def summary(self) -> dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn

        def div(a: float, b: float) -> float:
            return float(a / b) if b else 0.0

        precision = div(tp, tp + fp)
        recall = div(tp, tp + fn)
        return {
            "iou": div(tp, tp + fp + fn),
            "dice": div(2 * tp, 2 * tp + fp + fn),
            "pixel_acc": div(tp + tn, tp + fp + fn + tn),
            "precision": precision,
            "recall": recall,
            "f1": div(2 * precision * recall, precision + recall),
        }


def iou_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """IoU of each image (B x 1 x H x W). An image with no object and no prediction scores 1."""
    pred = pred.bool().flatten(1)
    target = target.bool().flatten(1)
    inter = (pred & target).sum(1).float()
    union = (pred | target).sum(1).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))


@torch.no_grad()
def collect_probs(model: torch.nn.Module, loader, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid probabilities and targets for a whole loader (kept small: uint8 targets)."""
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device))).cpu())
        targets.append(y.to(torch.uint8))
    return torch.cat(probs), torch.cat(targets)


def metrics_at(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    pred = probs > threshold
    conf = Confusion()
    conf.update(pred, targets)
    out = conf.summary()
    out["mean_iou_per_image"] = float(iou_per_image(pred, targets).mean())
    out["threshold"] = threshold
    return out


def threshold_sweep(
    probs: torch.Tensor, targets: torch.Tensor, thresholds: list[float] | None = None
) -> list[dict[str, float]]:
    """Metrics for each threshold in (0, 1). The notebook used range(40, 85, 5), i.e. 40..80."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 0.91, 0.05)]
    for t in thresholds:
        if not 0.0 < t < 1.0:
            raise ValueError(f"thresholds are probabilities in (0, 1), got {t}")
    return [metrics_at(probs, targets, t) for t in thresholds]
