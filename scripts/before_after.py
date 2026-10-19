"""Before / after figures: what the model predicts untrained, half-way and at the end.

Two steps, so the figures can be redrawn without training again:

    # 1. train a U-Net and record its predictions on N fixed test images after every epoch
    #    (epoch 0 = the untrained model) -> runs/progress/<name>/progress.npz
    python scripts/before_after.py train --name shapes --data data/shapes --size 96 --epochs 6 --max-train 600
    python scripts/before_after.py train --name pets --data data/pets --size 96 --epochs 6 --max-train 600

    # 2. draw the figures into assets/ (no training)
    python scripts/before_after.py plot

`plot` writes, for each recorded dataset:
  assets/before_after_<name>.png     image | true mask | epoch 0 | middle epoch | last epoch
  assets/progress_<name>.gif         the predicted mask over the image, one frame per epoch,
                                     with the epoch and the mean IoU in the caption
and, when runs/<name>/{unet,segnet,fcn}/best.pt exist (scripts/compare_all.py):
  assets/unet_segnet_fcn_<name>.png  the same images through U-Net, SegNet-like and the fully
                                     connected model
Overlay colours: green = correct object pixels, red = false positives, blue = missed pixels.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from unetseg.config import ModelConfig, TrainConfig  # noqa: E402
from unetseg.data import SegDataset, denormalize  # noqa: E402
from unetseg.train import load_checkpoint, train  # noqa: E402
from unetseg.viz import error_overlay  # noqa: E402

ASSETS = Path("assets")


def iou(pred: np.ndarray, true: np.ndarray) -> float:
    union = (pred | true).sum()
    return float((pred & true).sum() / union) if union else 1.0


def cmd_train(a) -> None:
    out = Path("runs/progress") / a.name
    out.mkdir(parents=True, exist_ok=True)
    test = SegDataset.from_folder(a.data, "test", a.size)
    idx = list(range(a.n_images))
    xs = torch.stack([test[i][0] for i in idx])
    probs = []

    @torch.no_grad()
    def record(epoch, model):
        was_training = model.training
        model.eval()
        p = torch.sigmoid(model(xs)).squeeze(1).numpy()
        probs.append((p * 255).round().astype(np.uint8))
        model.train(was_training)
        print(f"recorded epoch {epoch}", flush=True)

    tcfg = TrainConfig(data_dir=a.data, out_dir=str(out / a.model), img_size=a.size, epochs=a.epochs,
                       max_train=a.max_train, threads=a.threads, lr=a.lr)
    train(ModelConfig.from_preset(a.model), tcfg, on_model=record)
    np.savez_compressed(
        out / "progress.npz",
        probs=np.stack(probs),  # (epochs + 1) x N x H x W, uint8 probability * 255
        images=np.stack([(denormalize(x) * 255).round().astype(np.uint8) for x in xs]),
        masks=np.stack([test[i][1][0].numpy().astype(np.uint8) for i in idx]),
    )
    (out / "info.json").write_text(json.dumps({"model": a.model, "data": a.data, "size": a.size,
                                               "epochs": a.epochs, "max_train": a.max_train}, indent=1))
    print(f"wrote {out / 'progress.npz'}")


def _overlay(img_u8: np.ndarray, prob_u8: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    pred = (prob_u8 > 127).astype(np.uint8)
    return error_overlay(img_u8 / 255.0, pred, mask), iou(pred.astype(bool), mask.astype(bool))


def before_after(name: str, d: dict, model: str) -> None:
    probs, images, masks = d["probs"], d["images"], d["masks"]
    last = len(probs) - 1
    epochs = [0, last // 2, last]
    n = len(images)
    fig, ax = plt.subplots(n, 5, figsize=(10, 2.1 * n))
    for i in range(n):
        ax[i, 0].imshow(images[i])
        ax[i, 1].imshow(masks[i], cmap="gray")
        for j, e in enumerate(epochs):
            ov, v = _overlay(images[i], probs[e, i], masks[i])
            ax[i, 2 + j].imshow(ov)
            ax[i, 2 + j].set_xlabel(f"IoU {v:.2f}", fontsize=9)
    titles = ["image", "true mask", "epoch 0 (untrained)", f"epoch {epochs[1]}", f"epoch {last} (final)"]
    for j, t in enumerate(titles):
        ax[0, j].set_title(t, fontsize=10)
    for x in ax.ravel():
        x.set_xticks([])
        x.set_yticks([])
    fig.suptitle(f"{model} on {name}: predicted mask before, during and after training "
                 f"(green = correct, red = false positive, blue = missed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ASSETS / f"before_after_{name}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def progress_gif(name: str, d: dict, model: str, ms_per_frame: int = 900) -> None:
    probs, images, masks = d["probs"], d["images"], d["masks"]
    n = len(images)
    frames = []
    for e in range(len(probs)):
        fig, ax = plt.subplots(1, n, figsize=(2.3 * n, 2.75))
        ious = []
        for i in range(n):
            ov, v = _overlay(images[i], probs[e, i], masks[i])
            ious.append(v)
            ax[i].imshow(ov)
            ax[i].set_xlabel(f"IoU {v:.2f}", fontsize=9)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        label = "untrained" if e == 0 else f"epoch {e}/{len(probs) - 1}"
        fig.suptitle(f"{model} on {name}, {label}: mean IoU {np.mean(ious):.2f}", fontsize=11)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB").quantize(colors=128))
    durations = [ms_per_frame] * (len(frames) - 1) + [ms_per_frame * 3]  # hold the final frame
    path = ASSETS / f"progress_{name}.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


@torch.no_grad()
def model_panel(name: str, d: dict) -> None:
    run = Path("runs") / name
    names = [m for m in ("unet", "segnet", "fcn") if (run / m / "best.pt").exists()]
    if len(names) < 2:
        print(f"skip unet_segnet_fcn_{name}.png: needs runs/{name}/<model>/best.pt")
        return
    images, masks = d["images"], d["masks"]
    n = len(images)
    fig, ax = plt.subplots(n, 2 + len(names), figsize=(2.1 * (2 + len(names)), 2.1 * n))
    loaded = {m: load_checkpoint(run / m / "best.pt") for m in names}
    from unetseg.data import normalize

    for i in range(n):
        ax[i, 0].imshow(images[i])
        ax[i, 1].imshow(masks[i], cmap="gray")
        x = normalize(torch.from_numpy(images[i]).permute(2, 0, 1))[None]
        for j, m in enumerate(names):
            model, _, ckpt = loaded[m]
            size = ckpt["train_config"]["img_size"]
            if x.shape[-1] != size:
                raise ValueError(f"{m} was trained at {size}px, the progress images are {x.shape[-1]}px")
            prob = (torch.sigmoid(model(x))[0, 0].numpy() * 255).astype(np.uint8)
            ov, v = _overlay(images[i], prob, masks[i])
            ax[i, 2 + j].imshow(ov)
            ax[i, 2 + j].set_xlabel(f"IoU {v:.2f}", fontsize=9)
    labels = {"unet": "U-Net (skips)", "segnet": "SegNet-like (no skips)", "fcn": "fully connected"}
    for j, t in enumerate(["image", "true mask"] + [labels[m] for m in names]):
        ax[0, j].set_title(t, fontsize=10)
    for x_ in ax.ravel():
        x_.set_xticks([])
        x_.set_yticks([])
    fig.suptitle(f"same {name} test images, best checkpoints after 6 epochs", fontsize=10)
    fig.tight_layout()
    fig.savefig(ASSETS / f"unet_segnet_fcn_{name}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {ASSETS / f'unet_segnet_fcn_{name}.png'}")


def cmd_plot(a) -> None:
    ASSETS.mkdir(exist_ok=True)
    for f in sorted(Path("runs/progress").glob("*/progress.npz")):
        name = f.parent.name
        d = dict(np.load(f))
        model = json.loads((f.parent / "info.json").read_text())["model"]
        before_after(name, d, model)
        progress_gif(name, d, model)
        model_panel(name, d)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--model", default="unet")
    t.add_argument("--size", type=int, default=96)
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--max-train", type=int, default=600)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--n-images", type=int, default=4)
    t.add_argument("--threads", type=int, default=1)
    sub.add_parser("plot")
    a = p.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_plot(a)


if __name__ == "__main__":
    main()
