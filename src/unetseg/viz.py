"""Figures: training curves, prediction grids with error overlays, threshold sweep,
confusion matrix. Everything is written to PNG (matplotlib Agg), so it works on a
headless machine; in a notebook the same functions also return the figure.

The error overlay colours each pixel by its confusion-matrix cell:
green = true positive, red = false positive, blue = false negative, image = true negative.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from unetseg.data import denormalize  # noqa: E402


def _save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110, bbox_inches="tight")
    return fig


def plot_curves(history: list[dict], path=None, title: str = ""):
    """Loss, IoU/Dice and update ratio per epoch (the notebook's three-panel live plot)."""
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], "o-", label="val")
    ax[0].set(title="loss", xlabel="epoch")
    ax[1].plot(ep, [h["val_iou"] for h in history], "o-", label="val IoU")
    ax[1].plot(ep, [h["val_dice"] for h in history], "s--", label="val Dice")
    ax[1].plot(ep, [h["val_pixel_acc"] for h in history], "x:", label="val pixel acc", alpha=0.6)
    ax[1].set(title="validation metrics", xlabel="epoch", ylim=(0, 1))
    ax[2].semilogy(ep, [h["update_ratio"] for h in history], "^-", color="tab:green", label="||dW|| / ||W||")
    ax[2].axhline(1e-3, ls="--", color="tab:red", label="1e-3 reference")
    ax[2].set(title="parameter update ratio", xlabel="epoch")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)
    return fig


def error_overlay(image: np.ndarray, pred: np.ndarray, target: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """H x W x 3 image in [0,1] with TP green, FP red, FN blue blended on top."""
    out = image.copy()
    colours = {(1, 1): (0.1, 0.85, 0.2), (1, 0): (0.95, 0.15, 0.1), (0, 1): (0.15, 0.35, 1.0)}
    for (p, t), c in colours.items():
        sel = (pred == p) & (target == t)
        out[sel] = (1 - alpha) * out[sel] + alpha * np.array(c)
    return out


@torch.no_grad()
def prediction_grid(model, dataset, path=None, n: int = 6, threshold: float = 0.5, title: str = "", device="cpu"):
    """Rows: image | true mask | probability | error overlay, for the first n samples."""
    model.eval()
    n = min(n, len(dataset))
    fig, ax = plt.subplots(4, n, figsize=(2.2 * n, 9))
    ax = np.asarray(ax).reshape(4, n)
    for j in range(n):
        x, y = dataset[j]
        prob = torch.sigmoid(model(x[None].to(device)))[0, 0].cpu().numpy()
        img = denormalize(x)
        tgt = y[0].numpy().astype(np.uint8)
        pred = (prob > threshold).astype(np.uint8)
        inter, union = (pred & tgt).sum(), (pred | tgt).sum()
        iou = inter / union if union else 1.0
        ax[0, j].imshow(img)
        ax[1, j].imshow(tgt, cmap="gray", vmin=0, vmax=1)
        ax[2, j].imshow(prob, cmap="magma", vmin=0, vmax=1)
        ax[3, j].imshow(error_overlay(img, pred, tgt))
        ax[3, j].set_xlabel(f"IoU {iou:.2f}")
    for i, name in enumerate(["image", "true mask", "p(object)", "TP / FP / FN"]):
        ax[i, 0].set_ylabel(name)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)
    return fig


def plot_threshold_sweep(sweep: list[dict], path=None, title: str = "threshold sweep"):
    t = [s["threshold"] for s in sweep]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("iou", "dice", "precision", "recall"):
        ax.plot(t, [s[key] for s in sweep], "o-", label=key, ms=3)
    best = max(sweep, key=lambda s: s["iou"])
    ax.axvline(best["threshold"], ls="--", color="gray", label=f"best IoU @ {best['threshold']:.2f}")
    ax.set(xlabel="threshold on sigmoid(logit)", ylim=(0, 1.02), title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)
    return fig


def plot_confusion(matrix: np.ndarray, path=None, title: str = "pixel confusion matrix"):
    frac = matrix / max(matrix.sum(), 1)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=frac.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}\n{100 * frac[i, j]:.1f}%", ha="center", va="center",
                    color="white" if frac[i, j] > frac.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["bg", "object"], yticklabels=["bg", "object"],
           xlabel="predicted", ylabel="true", title=title)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)
    return fig
