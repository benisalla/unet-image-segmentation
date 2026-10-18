"""Step 6: one experiment from the coursework, small enough for a CPU: skip connections.

Trains `unet` (with skips) and `segnet` (same encoder-decoder, no skips) for 3 epochs
each on the example data (~4-6 minutes in total), then draws the same test images
through both. Without skips the decoder must rebuild the edges from the 6x6 bottleneck,
so the masks come out blurry and rounded. Writes examples/output/06_skip_vs_no_skip.png.
Needs example 01.

    python examples/06_skip_vs_no_skip.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from unetseg.config import ModelConfig, TrainConfig  # noqa: E402
from unetseg.data import SegDataset, denormalize  # noqa: E402
from unetseg.train import load_checkpoint, train  # noqa: E402

OUT = Path("examples/output")
OUT.mkdir(parents=True, exist_ok=True)
results = {}
for name in ["unet", "segnet"]:
    tcfg = TrainConfig(data_dir="data/examples/shapes", out_dir=f"runs/examples/{name}_3ep", img_size=96, epochs=3)
    s = train(ModelConfig.from_preset(name), tcfg, verbose=False)
    results[name] = s["test"]["at_threshold"]
    print(f"{name:7s} test IoU {results[name]['iou']:.3f}  Dice {results[name]['dice']:.3f}  "
          f"({s['params'] / 1e6:.2f} M params, {s['train_time_s'] / 60:.1f} min)")

test = SegDataset.from_folder("data/examples/shapes", "test", 96)
models = {n: load_checkpoint(f"runs/examples/{n}_3ep/best.pt")[0] for n in results}
fig, ax = plt.subplots(4, 4, figsize=(8, 8))
with torch.no_grad():
    for i in range(4):
        x, y = test[i]
        ax[i, 0].imshow(denormalize(x))
        ax[i, 1].imshow(y[0], cmap="gray")
        for j, m in enumerate(models.values()):
            ax[i, 2 + j].imshow(torch.sigmoid(m(x[None]))[0, 0].numpy(), cmap="magma", vmin=0, vmax=1)
for a, t in zip(ax[0], ["image", "true mask", f"unet p (IoU {results['unet']['iou']:.2f})",
                        f"segnet p (IoU {results['segnet']['iou']:.2f})"], strict=True):
    a.set_title(t, fontsize=9)
for a in ax.ravel():
    a.axis("off")
fig.tight_layout()
fig.savefig(OUT / "06_skip_vs_no_skip.png", dpi=100)
print(f"wrote {OUT / '06_skip_vs_no_skip.png'}  (unet - segnet IoU: "
      f"{results['unet']['iou'] - results['segnet']['iou']:+.3f})")
