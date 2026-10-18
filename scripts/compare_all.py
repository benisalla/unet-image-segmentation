"""Train several variants one after another on the same data and write a comparison.

    python scripts/compare_all.py --data data/shapes --models unet unet-strided unet-single \
        unet-residual segnet fcn --epochs 10 --out runs/shapes

Writes <out>/<model>/ (one run folder each, see unetseg-train), <out>/results.md (table),
<out>/results.json, <out>/compare_iou.png (val IoU per epoch for every model) and
<out>/compare_predictions.png (the same test images through every model).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from unetseg.config import ModelConfig, TrainConfig  # noqa: E402
from unetseg.data import SegDataset, denormalize  # noqa: E402
from unetseg.train import load_checkpoint, train  # noqa: E402
from unetseg.viz import error_overlay  # noqa: E402


def results_table(rows: list[dict]) -> str:
    lines = ["| model | params | epochs | test IoU | test Dice | pixel acc | precision | recall | "
             "best val threshold | IoU at it | train time |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["test"]["at_threshold"]
        b = r["test"]["at_best_threshold"]
        lines.append(
            f"| `{r['model']}` | {r['params'] / 1e6:.2f} M | {r['epochs']} | {t['iou']:.3f} | {t['dice']:.3f} | "
            f"{t['pixel_acc']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} | "
            f"{r['test']['best_val_threshold']:.2f} | {b['iou']:.3f} | {r['train_time_s'] / 60:.1f} min |")
    return "\n".join(lines) + "\n"


@torch.no_grad()
def compare_predictions(out: Path, data: str, models: list[str], n: int = 5) -> None:
    loaded = [(m, *load_checkpoint(out / m / "best.pt")[::2]) for m in models]
    size = loaded[0][2]["train_config"]["img_size"]
    ds = SegDataset.from_folder(data, "test", size)
    fig, ax = plt.subplots(n, len(models) + 2, figsize=(2.1 * (len(models) + 2), 2.2 * n))
    for i in range(n):
        x, y = ds[i]
        img, tgt = denormalize(x), y[0].numpy().astype(np.uint8)
        ax[i, 0].imshow(img)
        ax[i, 1].imshow(tgt, cmap="gray")
        for j, (name, model, _) in enumerate(loaded):
            pred = (torch.sigmoid(model(x[None]))[0, 0] > 0.5).numpy().astype(np.uint8)
            union = (pred | tgt).sum()
            ax[i, j + 2].imshow(error_overlay(img, pred, tgt))
            ax[i, j + 2].set_xlabel(f"IoU {(pred & tgt).sum() / union if union else 1:.2f}", fontsize=8)
            if i == 0:
                ax[0, j + 2].set_title(name, fontsize=9)
    ax[0, 0].set_title("image", fontsize=9)
    ax[0, 1].set_title("true mask", fontsize=9)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("test images through every model (green TP, red FP, blue FN)")
    fig.tight_layout()
    fig.savefig(out / "compare_predictions.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/shapes")
    p.add_argument("--models", nargs="+", default=["unet", "unet-strided", "unet-single", "unet-residual",
                                                   "segnet", "fcn"])
    p.add_argument("--out", default="runs/shapes")
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--loss", default="bce+dice")
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--plot-only", action="store_true", help="only rebuild the tables/figures")
    a = p.parse_args()
    out = Path(a.out)
    rows = []
    for m in a.models:
        if not a.plot_only:
            mcfg = ModelConfig.from_preset(m, base=a.base)
            tcfg = TrainConfig(data_dir=a.data, out_dir=str(out / m), img_size=a.size, epochs=a.epochs,
                               batch_size=a.batch_size, lr=a.lr, loss=a.loss, threads=a.threads,
                               max_train=a.max_train)
            print(f"\n=== {m} ===")
            train(mcfg, tcfg)
        rows.append(json.loads((out / m / "summary.json").read_text()))
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    (out / "results.md").write_text(results_table(rows))
    print(results_table(rows))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for m in a.models:
        h = json.loads((out / m / "history.json").read_text())
        ax[0].plot([r["epoch"] for r in h], [r["val_iou"] for r in h], "o-", ms=3, label=m)
        ax[1].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], "o-", ms=3, label=m)
    ax[0].set(title="validation IoU", xlabel="epoch", ylim=(0, 1))
    ax[1].set(title="training loss", xlabel="epoch")
    for x in ax:
        x.grid(alpha=0.3)
        x.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "compare_iou.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    compare_predictions(out, a.data, a.models)


if __name__ == "__main__":
    main()
