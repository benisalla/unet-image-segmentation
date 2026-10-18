"""Step 4: segment image files with the trained model and draw overlays.

Uses the same function as `unetseg-predict`: resize to the training size, normalise,
predict, resize the probability map back, threshold. Writes
examples/output/04_predictions.png (image | prediction overlay | TP/FP/FN). Needs 02.

    python examples/04_predict_and_overlay.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from unetseg.predict import overlay, predict_image  # noqa: E402
from unetseg.train import load_checkpoint  # noqa: E402
from unetseg.viz import error_overlay  # noqa: E402

OUT = Path("examples/output")
OUT.mkdir(parents=True, exist_ok=True)
model, cfg, ckpt = load_checkpoint("runs/examples/unet/best.pt")
size = ckpt["train_config"]["img_size"]
files = sorted(Path("data/examples/shapes/test/images").glob("*.png"))[:4]

fig, ax = plt.subplots(len(files), 3, figsize=(7, 2.4 * len(files)))
for i, f in enumerate(files):
    img = Image.open(f).convert("RGB")
    pred = predict_image(model, img, size) > 0.5
    true = np.asarray(Image.open(f.parent.parent / "masks" / f.name)) > 0
    iou = (pred & true).sum() / max((pred | true).sum(), 1)
    ax[i, 0].imshow(img)
    ax[i, 1].imshow(overlay(img, pred))
    ax[i, 2].imshow(error_overlay(np.asarray(img) / 255.0, pred.astype(np.uint8), true.astype(np.uint8)))
    ax[i, 2].set_title(f"IoU {iou:.2f}", fontsize=9)
    print(f"{f.name}: {100 * pred.mean():.1f}% predicted object, IoU {iou:.3f}")
for a, t in zip(ax[0], ["image", "prediction", "green TP / red FP / blue FN"], strict=True):
    a.set_title(t, fontsize=9)
for a in ax.ravel():
    a.axis("off")
fig.tight_layout()
fig.savefig(OUT / "04_predictions.png", dpi=100)
print(f"wrote {OUT / '04_predictions.png'}")
