"""`unetseg-train`: train one model variant and evaluate it.

The loop, per epoch:
  1. train on every batch: forward -> loss (BCE / Dice / both) -> backward ->
     gradient clipping -> AdamW step -> cosine learning-rate step;
  2. validate at `--threshold`: loss, IoU, Dice, pixel accuracy;
  3. keep `best.pt` by **validation IoU** (not pixel accuracy, which a model that
     predicts only background already scores high on), always write `last.pt`,
     `history.json` and `curves.png` (redrawn every epoch, so it is a live plot).
At the end the best checkpoint is evaluated on the test split (see `evaluate.py`).

The update ratio ||W_after - W_before|| / ||W_before||, averaged over the parameter
tensors, is logged as in the notebook; around 1e-3 is the usual healthy range.
It is measured after the AdamW step (the notebook used lr * grad, which is not the
step Adam takes).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from unetseg.config import PRESETS, ModelConfig, TrainConfig
from unetseg.data import make_loaders
from unetseg.losses import SegLoss
from unetseg.metrics import Confusion
from unetseg.models import build_model, count_parameters
from unetseg.viz import plot_curves


def set_threads(n: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    torch.set_num_threads(n)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def validate(model, loader, criterion, threshold=0.5, device="cpu") -> dict[str, float]:
    model.eval()
    conf, total, n = Confusion(), 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits, y).item() * len(x)
        n += len(x)
        conf.update(torch.sigmoid(logits) > threshold, y > 0.5)
    out = conf.summary()
    out["loss"] = total / max(n, 1)
    return out


def train_one_epoch(model, loader, criterion, optimizer, scheduler, max_norm=None, device="cpu"):
    model.train()
    total, n, ratios = 0.0, 0, []
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        track = step % 10 == 0
        if track:
            before = [p.detach().clone() for p in model.parameters()]
        optimizer.step()
        scheduler.step()
        if track:
            # skip tensors that are exactly zero (fresh biases): their ratio is undefined
            pairs = zip(model.parameters(), before, strict=True)
            r = [((p.detach() - b).norm() / b.norm()).item() for p, b in pairs if b.norm() > 0]
            ratios.append(float(np.mean(r)) if r else 0.0)
        total += loss.item() * len(x)
        n += len(x)
    return total / max(n, 1), float(np.mean(ratios)) if ratios else 0.0


def save_checkpoint(path, model, mcfg: ModelConfig, tcfg: TrainConfig, epoch: int, metrics: dict) -> None:
    torch.save(
        {"model": model.state_dict(), "model_config": asdict(mcfg), "train_config": tcfg.to_dict(),
         "epoch": epoch, "val": metrics},
        path,
    )


def load_checkpoint(path, device="cpu"):
    """-> (model in eval mode, ModelConfig, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = build_model(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, mcfg, ckpt


def train(mcfg: ModelConfig, tcfg: TrainConfig, evaluate_test: bool = True, verbose: bool = True,
          on_epoch=None) -> dict:
    """Train, keep the best checkpoint by val IoU, evaluate it. Returns a summary dict.

    `on_epoch(history)` is called after every epoch (the notebook uses it to redraw the
    curves in place).
    """
    set_threads(tcfg.threads)
    seed_everything(tcfg.seed)
    out = Path(tcfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mcfg.img_size = tcfg.img_size

    train_dl, val_dl, test_dl = make_loaders(tcfg.data_dir, tcfg.img_size, tcfg.batch_size, tcfg.augment,
                                             tcfg.max_train, tcfg.seed)
    model = build_model(mcfg).to(tcfg.device)
    n_params = count_parameters(model)
    criterion = SegLoss(tcfg.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    total_steps = max(1, tcfg.epochs * len(train_dl))
    warmup = min(50, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / max(1, warmup) if s < warmup
        else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
    )
    if verbose:
        fg = train_dl.dataset.foreground_fraction()
        print(f"model {mcfg.name}: {n_params / 1e6:.2f} M parameters | train {len(train_dl.dataset)} "
              f"val {len(val_dl.dataset)} | foreground {100 * fg:.1f}% of pixels | loss {tcfg.loss}")

    history, best_iou, t0 = [], -1.0, time.time()
    for epoch in range(1, tcfg.epochs + 1):
        te = time.time()
        tr_loss, ratio = train_one_epoch(model, train_dl, criterion, optimizer, scheduler, tcfg.max_norm, tcfg.device)
        val = validate(model, val_dl, criterion, tcfg.threshold, tcfg.device)
        row = {"epoch": epoch, "train_loss": tr_loss, "update_ratio": ratio,
               "lr": optimizer.param_groups[0]["lr"], "time_s": time.time() - te,
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        improved = val["iou"] > best_iou
        if improved:
            best_iou = val["iou"]
            save_checkpoint(out / "best.pt", model, mcfg, tcfg, epoch, val)
        save_checkpoint(out / "last.pt", model, mcfg, tcfg, epoch, val)
        (out / "history.json").write_text(json.dumps(history, indent=1))
        plot_curves(history, out / "curves.png", title=f"{mcfg.name} - epoch {epoch}/{tcfg.epochs}")
        if verbose:
            print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} | val loss {val['loss']:.4f} "
                  f"IoU {val['iou']:.4f} Dice {val['dice']:.4f} acc {val['pixel_acc']:.4f} | "
                  f"upd {ratio:.1e} | {row['time_s']:.0f}s{'  *best' if improved else ''}", flush=True)
        if on_epoch is not None:
            on_epoch(history)

    summary = {"model": mcfg.name, "params": n_params, "epochs": tcfg.epochs, "train_time_s": time.time() - t0,
               "best_val_iou": best_iou, "model_config": asdict(mcfg), "train_config": tcfg.to_dict()}
    if evaluate_test:
        from unetseg.evaluate import evaluate_checkpoint

        summary["test"] = evaluate_checkpoint(out / "best.pt", tcfg.data_dir, out, split="test",
                                              threshold=tcfg.threshold, verbose=verbose)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="unetseg-train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="unet", choices=sorted(PRESETS), help="architecture preset")
    p.add_argument("--data", default="data/shapes", help="folder with train/ val/ test/")
    p.add_argument("--out", default=None, help="run folder (default runs/<model>)")
    p.add_argument("--size", type=int, default=96, help="input size (divisible by 2**depth)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", default="bce+dice", choices=["bce", "dice", "bce+dice"])
    p.add_argument("--augment", default="torchvision", choices=["none", "torchvision", "albumentations"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-train", type=int, default=None, help="use only the first N training images")
    p.add_argument("--base", type=int, default=16, help="channels of the first block")
    p.add_argument("--depth", type=int, default=4, help="number of down-sampling steps")
    g = p.add_argument_group("switches (override the preset)")
    g.add_argument("--skip", action=argparse.BooleanOptionalAction, default=None, help="U-Net skip connections")
    g.add_argument("--down", choices=["maxpool", "strided"], default=None)
    g.add_argument("--double", action=argparse.BooleanOptionalAction, default=None, help="two convs per block")
    g.add_argument("--residual", action=argparse.BooleanOptionalAction, default=None, help="residual blocks")
    p.add_argument("--fcn-hidden", type=int, default=256)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    overrides = dict(skip=a.skip, down=a.down, double=a.double, residual=a.residual) if a.model != "fcn" else {}
    mcfg = ModelConfig.from_preset(a.model, base=a.base, depth=a.depth, fcn_hidden=a.fcn_hidden, **overrides)
    tcfg = TrainConfig(data_dir=a.data, out_dir=a.out or f"runs/{a.model}", img_size=a.size, epochs=a.epochs,
                       batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay, loss=a.loss,
                       threshold=a.threshold, augment=a.augment, max_train=a.max_train, threads=a.threads,
                       seed=a.seed, device=a.device)
    train(mcfg, tcfg)


if __name__ == "__main__":
    main()
