"""`unetseg-predict`: segment image files with a trained checkpoint.

For each input image it writes `<stem>_mask.png` (0/255 at the original resolution)
and `<stem>_overlay.png` (the predicted object tinted green on the image). The image
is resized to the training size, normalised, passed through the model, and the
probability map is resized back before thresholding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from unetseg.data import normalize


@torch.no_grad()
def predict_image(model, image: Image.Image, size: int, threshold: float = 0.5, device="cpu") -> np.ndarray:
    """PIL image -> H x W float probability map at the image's own resolution."""
    rgb = image.convert("RGB")
    x = torch.from_numpy(np.asarray(rgb.resize((size, size), Image.BILINEAR)).copy()).permute(2, 0, 1)
    logits = model(normalize(x)[None].to(device))
    prob = torch.sigmoid(F.interpolate(logits, size=(rgb.height, rgb.width), mode="bilinear", align_corners=False))
    return prob[0, 0].cpu().numpy()


def overlay(image: Image.Image, mask: np.ndarray, colour=(30, 220, 60), alpha: float = 0.5) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    arr[mask] = (1 - alpha) * arr[mask] + alpha * np.array(colour, dtype=np.float32)
    return Image.fromarray(arr.astype(np.uint8))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="unetseg-predict", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("images", nargs="+")
    p.add_argument("--out", default="predictions")
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args(argv)
    from unetseg.train import load_checkpoint

    model, _, ckpt = load_checkpoint(a.checkpoint)
    size = ckpt["train_config"]["img_size"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in a.images:
        img = Image.open(path)
        mask = predict_image(model, img, size, a.threshold) > a.threshold
        stem = Path(path).stem
        Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"{stem}_mask.png")
        overlay(img, mask).save(out / f"{stem}_overlay.png")
        print(f"{path}: {100 * mask.mean():.1f}% of pixels predicted object -> {out}/{stem}_overlay.png")


if __name__ == "__main__":
    main()
