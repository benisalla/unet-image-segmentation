"""Step 1: write a small synthetic shapes dataset and look at it, with augmentations.

Writes data/examples/shapes (400 train / 60 val / 60 test, 96x96, ~3 s) and
examples/output/01_samples.png: images, their masks, and the same image/mask pair
after four random augmentations (image and mask always move together).

    python examples/01_make_data_and_look.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from unetseg.data import SegDataset, TorchvisionAugment  # noqa: E402
from unetseg.datasets import write_shapes  # noqa: E402

DATA = Path("data/examples/shapes")
OUT = Path("examples/output")
OUT.mkdir(parents=True, exist_ok=True)

write_shapes(DATA, size=96, n_train=400, n_val=60, n_test=60, seed=0)
ds = SegDataset.from_folder(DATA, "train", 96)
print(f"{len(ds)} training images, object = {100 * ds.foreground_fraction():.1f}% of the pixels")

torch.manual_seed(0)
aug = TorchvisionAugment()
fig, ax = plt.subplots(2, 6, figsize=(13, 4.6))
for j in range(2):  # two raw samples
    ax[0, j].imshow(ds.images[j].permute(1, 2, 0))
    ax[1, j].imshow(ds.masks[j, 0], cmap="gray")
    ax[0, j].set_title(f"sample {j}")
for j in range(2, 6):  # sample 0 augmented four times
    img, mask = aug(ds.images[0], ds.masks[0])
    ax[0, j].imshow(img.permute(1, 2, 0))
    ax[1, j].imshow(mask[0], cmap="gray")
    ax[0, j].set_title(f"sample 0, aug {j - 1}")
for a in ax.ravel():
    a.axis("off")
fig.tight_layout()
fig.savefig(OUT / "01_samples.png", dpi=100)
print(f"wrote {OUT / '01_samples.png'}")
