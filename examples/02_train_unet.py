"""Step 2: train the default U-Net (skip connections, max-pool, double conv) for 8 epochs.

About 3-4 minutes on one CPU thread (test IoU ~0.7 on the 64 px example shapes).
Writes runs/examples/unet/ (best.pt, curves.png,
history.json, metrics.json, predictions.png ...). Needs example 01.

    python examples/02_train_unet.py
"""

from unetseg.config import ModelConfig, TrainConfig
from unetseg.train import train

mcfg = ModelConfig.from_preset("unet", base=16, depth=4)
tcfg = TrainConfig(
    data_dir="data/examples/shapes",
    out_dir="runs/examples/unet",
    img_size=64,
    epochs=8,
    batch_size=16,
    lr=3e-3,
    loss="bce+dice",
    threads=1,
)
summary = train(mcfg, tcfg)
test = summary["test"]["at_threshold"]
print(f"\ntest IoU {test['iou']:.3f}  Dice {test['dice']:.3f}  pixel acc {test['pixel_acc']:.3f}")
print("curves: runs/examples/unet/curves.png   predictions: runs/examples/unet/predictions.png")
