"""Step 3: evaluate the checkpoint of example 02 and see what the threshold does.

Collects sigmoid probabilities on the validation split once, then scores them at
thresholds 0.1 ... 0.9: low thresholds raise recall (more pixels called object),
high thresholds raise precision. Writes examples/output/03_thresholds.png and
03_confusion.png. Needs example 02.

    python examples/03_evaluate_and_thresholds.py
"""

from pathlib import Path

from torch.utils.data import DataLoader

from unetseg.data import SegDataset
from unetseg.metrics import Confusion, collect_probs, threshold_sweep
from unetseg.train import load_checkpoint
from unetseg.viz import plot_confusion, plot_threshold_sweep

OUT = Path("examples/output")
OUT.mkdir(parents=True, exist_ok=True)
model, cfg, ckpt = load_checkpoint("runs/examples/unet/best.pt")
val = SegDataset.from_folder("data/examples/shapes", "val", ckpt["train_config"]["img_size"])
probs, targets = collect_probs(model, DataLoader(val, batch_size=32))

print("threshold   IoU    Dice   precision recall")
sweep = threshold_sweep(probs, targets)
for s in sweep:
    print(f"   {s['threshold']:.2f}    {s['iou']:.3f}  {s['dice']:.3f}   {s['precision']:.3f}    {s['recall']:.3f}")
best = max(sweep, key=lambda s: s["iou"])
print(f"best IoU {best['iou']:.3f} at threshold {best['threshold']:.2f}")
plot_threshold_sweep(sweep, OUT / "03_thresholds.png", title="unet on shapes (val)")

conf = Confusion()
conf.update(probs > 0.5, targets)
print("confusion matrix [[TN FP] [FN TP]]:", conf.matrix().tolist())
plot_confusion(conf.matrix(), OUT / "03_confusion.png")
print(f"wrote {OUT / '03_thresholds.png'} and {OUT / '03_confusion.png'}")
