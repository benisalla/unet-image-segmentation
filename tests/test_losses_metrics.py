import pytest
import torch

from unetseg.losses import SegLoss, soft_dice_loss
from unetseg.metrics import Confusion, iou_per_image, metrics_at, threshold_sweep


def test_dice_loss_is_zero_for_a_perfect_confident_prediction():
    y = torch.zeros(2, 1, 8, 8)
    y[:, :, 2:6, 2:6] = 1
    logits = (y * 2 - 1) * 50
    assert soft_dice_loss(logits, y) < 1e-3
    assert soft_dice_loss(-logits, y) > 0.9


def test_bce_plus_dice_is_the_sum():
    torch.manual_seed(0)
    z, y = torch.randn(2, 1, 8, 8), (torch.rand(2, 1, 8, 8) > 0.5).float()
    total = SegLoss("bce+dice")(z, y)
    assert torch.allclose(total, SegLoss("bce")(z, y) + SegLoss("dice")(z, y))
    with pytest.raises(ValueError):
        SegLoss("focal")


def test_confusion_metrics_by_hand():
    pred = torch.tensor([1, 1, 0, 0, 1, 0, 0, 0])
    true = torch.tensor([1, 0, 1, 0, 1, 0, 0, 0])
    c = Confusion()
    c.update(pred, true)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 4)
    s = c.summary()
    assert s["iou"] == pytest.approx(2 / 4)
    assert s["dice"] == pytest.approx(4 / 6)
    assert s["dice"] == pytest.approx(s["f1"])  # Dice is F1 on pixels
    assert s["pixel_acc"] == pytest.approx(6 / 8)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)
    assert c.matrix().tolist() == [[4, 1], [1, 2]]


def test_pixel_accuracy_rewards_predicting_nothing_iou_does_not():
    y = torch.zeros(1, 1, 10, 10)
    y[..., :2, :] = 1  # 20% object
    m = metrics_at(torch.zeros_like(y), y.to(torch.uint8), 0.5)
    assert m["pixel_acc"] == pytest.approx(0.8)
    assert m["iou"] == 0 and m["dice"] == 0


def test_empty_image_with_empty_prediction_scores_one():
    z = torch.zeros(1, 1, 4, 4)
    assert iou_per_image(z, z).item() == 1.0


def test_threshold_sweep_needs_probabilities():
    probs = torch.rand(2, 1, 8, 8)
    y = (probs > 0.6).to(torch.uint8)
    sweep = threshold_sweep(probs, y, [0.3, 0.6, 0.9])
    assert max(sweep, key=lambda s: s["iou"])["threshold"] == 0.6
    with pytest.raises(ValueError):
        threshold_sweep(probs, y, [40, 45])  # the notebook's range(40, 85, 5)
