"""`unetseg-eval`: score a checkpoint on a split.

Writes, next to the checkpoint (or in `--out`):
  metrics.json      IoU, Dice, pixel accuracy, precision, recall, F1, mean per-image IoU,
                    confusion matrix, at the training threshold and at the best threshold
  thresholds.png    IoU / Dice / precision / recall as the threshold moves from 0.1 to 0.9
  confusion.png     the pixel confusion matrix
  predictions.png   image | true mask | probability | TP/FP/FN overlay

The threshold is chosen on the **validation** split (picking it on the test split
would tune on the numbers you report); the test numbers are then given at both 0.5
and that threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from unetseg.data import SegDataset
from unetseg.metrics import Confusion, collect_probs, metrics_at, threshold_sweep


def evaluate_checkpoint(ckpt_path, data_dir, out_dir=None, split="test", threshold=0.5, batch_size=32,
                        device="cpu", verbose=True, n_show=6) -> dict:
    from unetseg.train import load_checkpoint
    from unetseg.viz import plot_confusion, plot_threshold_sweep, prediction_grid

    model, mcfg, ckpt = load_checkpoint(ckpt_path, device)
    size = ckpt["train_config"]["img_size"]
    out = Path(out_dir or Path(ckpt_path).parent)
    out.mkdir(parents=True, exist_ok=True)
    split_dir = Path(data_dir) / split
    if not split_dir.is_dir():
        split = "val"
    ds = SegDataset.from_folder(data_dir, split, size)
    val_ds = SegDataset.from_folder(data_dir, "val", size)

    val_probs, val_t = collect_probs(model, DataLoader(val_ds, batch_size=batch_size), device)
    sweep = threshold_sweep(val_probs, val_t)
    best_t = max(sweep, key=lambda s: s["iou"])["threshold"]

    probs, targets = collect_probs(model, DataLoader(ds, batch_size=batch_size), device)
    at_default = metrics_at(probs, targets, threshold)
    at_best = metrics_at(probs, targets, best_t)
    conf = Confusion()
    conf.update(probs > threshold, targets)
    result = {"split": split, "checkpoint": str(ckpt_path), "model": mcfg.name, "n_images": len(ds),
              "at_threshold": at_default, "best_val_threshold": best_t, "at_best_threshold": at_best,
              "confusion_matrix": conf.matrix().tolist(), "val_threshold_sweep": sweep}
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    plot_threshold_sweep(sweep, out / "thresholds.png", title=f"{mcfg.name}: threshold sweep (val)")
    plot_confusion(conf.matrix(), out / "confusion.png", title=f"{mcfg.name} ({split}, t={threshold})")
    prediction_grid(model, ds, out / "predictions.png", n=n_show, threshold=threshold,
                    title=f"{mcfg.name} on {split}: IoU {at_default['iou']:.3f}", device=device)
    if verbose:
        m = at_default
        print(f"[{split}] {len(ds)} images | t={threshold}: IoU {m['iou']:.4f} Dice {m['dice']:.4f} "
              f"acc {m['pixel_acc']:.4f} P {m['precision']:.4f} R {m['recall']:.4f} | "
              f"best val threshold {best_t:.2f}: test IoU {at_best['iou']:.4f}")
        print(f"confusion [[TN FP] [FN TP]] = {conf.matrix().tolist()}")
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="unetseg-eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--data", default=None, help="dataset folder (default: the one used for training)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default=None)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    data = a.data or torch.load(a.checkpoint, map_location="cpu", weights_only=False)["train_config"]["data_dir"]
    evaluate_checkpoint(a.checkpoint, data, a.out, a.split, a.threshold)


if __name__ == "__main__":
    main()
