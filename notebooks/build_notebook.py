"""Build notebooks/train_colab_kaggle.ipynb.

The notebook installs this repo with pip and imports `unetseg`, so it always runs the
same code as the CLIs. Code is shown with `inspect.getsource` where the explanation
needs it. Regenerate after changing the text below:

    python notebooks/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("train_colab_kaggle.ipynb")
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# U-Net image segmentation: every switch, explained and trained on Oxford-IIIT Pet

This notebook goes through binary segmentation with U-Net one idea at a time, and then
trains the models for real on a free GPU (Kaggle or Colab). It uses the `unetseg`
package of the [unet-image-segmentation](https://github.com/benisalla/unet-image-segmentation)
repo, so every cell runs exactly the code of the command-line tools.

The task: for every pixel, say whether it belongs to the object (here: the pet) or to
the background. The model outputs one number (a logit) per pixel; a sigmoid turns it
into a probability, and a threshold turns that into a mask.

**Sections:** 1 Install · 2 Data · 3 U-Net, piece by piece · 4 Losses and metrics ·
5 Train U-Net on Pets · 6 Compare every variant · 7 Thresholds · 8 Predictions ·
9 Conclusions · 10 Download

| | Kaggle | Colab |
|---|---|---|
| GPU | *Settings → Accelerator → GPU T4* | *Runtime → Change runtime type → T4 GPU* |
| Internet | *Settings → Internet on* (needed for pip and the dataset) | on by default |
| Outputs | `/kaggle/working` | `/content` (download them at the end) |

**Just checking it runs?** Set `SMOKE_TEST = True` in the config cell: it uses the
synthetic shapes dataset and one epoch per model, a few minutes even on a CPU.
""")

md("""
## 1. Install

The repo is private: on Kaggle add a secret `GITHUB_TOKEN` (*Add-ons → Secrets*), on Colab
add it in the key icon panel. With a public fork no token is needed.
Kaggle and Colab already have PyTorch, torchvision and matplotlib.
""")
code("""
import os, subprocess, sys

def github_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN")
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get("GITHUB_TOKEN")
    except Exception:
        return os.environ.get("GITHUB_TOKEN", "")

token = github_token()
auth = f"{token}@" if token else ""
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       f"git+https://{auth}github.com/benisalla/unet-image-segmentation.git"])
""")
code("""
import inspect, json, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Image as ShowImage, Markdown, clear_output, display

import unetseg
from unetseg import data, datasets, losses, metrics, models, viz
from unetseg.config import PRESETS, ModelConfig, TrainConfig
from unetseg.train import load_checkpoint, train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu")
""")
md("""
### Config

Real run (GPU): all 7 390 Pet photos at 160x160, a wider U-Net (`BASE = 32`, the notebook
of the coursework used 64), 25 epochs for U-Net and 12 for each variant in the comparison.
On a T4 this is about 40-60 minutes in total.
""")
code("""
SMOKE_TEST = False          # True: synthetic shapes, 1 epoch, runs on a CPU

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
WORK.mkdir(parents=True, exist_ok=True)
if SMOKE_TEST:
    DATA, SIZE, MAX_IMAGES, BASE, EPOCHS, CMP_EPOCHS, BATCH = WORK / "shapes", 64, 0, 8, 1, 1, 16
else:
    DATA, SIZE, MAX_IMAGES, BASE, EPOCHS, CMP_EPOCHS, BATCH = WORK / "pets", 160, 7390, 32, 25, 12, 32
LR, LOSS = 2e-3, "bce+dice"
""")

md("""
## 2. Data

**Oxford-IIIT Pet** (Parkhi et al. 2012): 37 cat and dog breeds, each photo comes with
a *trimap*: 1 = pet, 2 = background, 3 = the uncertain border. For binary segmentation the
border is counted as pet (it belongs to the animal's outline). `unetseg-data pets` streams
the 790 MB archive and keeps a random subset (here all of it), resized to `SIZE`, split
80 / 10 / 10 into train / val / test.

The coursework used a private competition dataset that cannot be shared; Pets is the
public replacement, and the synthetic shapes are the offline one.
""")
code("""
if SMOKE_TEST:
    datasets.write_shapes(DATA, size=SIZE, n_train=200, n_val=40, n_test=40)
elif not (DATA / "train").exists():
    datasets.write_pets(DATA, max_images=MAX_IMAGES, size=SIZE)
train_ds = data.SegDataset.from_folder(DATA, "train", SIZE)
print(len(train_ds), "train images | object =", f"{100 * train_ds.foreground_fraction():.1f}% of the pixels")

fig, ax = plt.subplots(2, 6, figsize=(14, 5))
for j in range(6):
    ax[0, j].imshow(train_ds.images[j].permute(1, 2, 0)); ax[1, j].imshow(train_ds.masks[j, 0], cmap="gray")
    ax[0, j].axis("off"); ax[1, j].axis("off")
plt.show()
""")
md("""
### Augmentation: image and mask move together

A flip or a rotation of the image must be applied to the mask too, with nearest-neighbour
interpolation for the mask (a mask pixel is 0 or 1, never 0.5). Only the training split
is augmented. (In the coursework notebook, validation and test also received the
training augmentations, because the attribute meant to switch them off had the wrong
name.)
""")
code("""
print(inspect.getsource(data.TorchvisionAugment))
aug = data.TorchvisionAugment()
fig, ax = plt.subplots(2, 5, figsize=(12, 5))
for j in range(5):
    im, mk = (train_ds.images[0], train_ds.masks[0]) if j == 0 else aug(train_ds.images[0], train_ds.masks[0])
    ax[0, j].imshow(im.permute(1, 2, 0)); ax[1, j].imshow(mk[0], cmap="gray")
    ax[0, j].set_title("original" if j == 0 else f"augmented {j}"); ax[0, j].axis("off"); ax[1, j].axis("off")
plt.show()
""")

md("""
## 3. U-Net, piece by piece

```
image ─ stem ─────────────────────────────────────────────── up1 ─ head ─ logits
          └ down1 ──────────────────────────────────── up2 ─┘
               └ down2 ─────────────────────── up3 ─┘
                    └ down3 ───────────── up4 ─┘
                         └ down4 (bottleneck)┘
```

**Encoder** (left, going down): each step halves the resolution and doubles the channels.
Lower resolution means each feature sees a larger part of the image: by the bottleneck a
6x6 (or 10x10) map summarises *what* is in the image. **Decoder** (right, going up): a
transposed convolution doubles the resolution again, back to one logit per pixel.

The problem: the bottleneck knows *what* but has lost *where*, the exact edges.
**Skip connections** (the horizontal lines) hand the decoder the encoder's feature map of
the same resolution, so it can copy the fine detail instead of guessing it. Without them
the model is a plain encoder-decoder (`segnet` here).

### 3.1 The convolution block: single or double conv, optional residual
""")
code("""
print(inspect.getsource(models.ConvBlock))
""")
md("""
* **double conv** (`double=True`, the U-Net paper): two 3x3 convs in a row see a 5x5 window,
  with fewer weights than one 5x5 conv and one more ReLU in between, so the block is more
  expressive at the same resolution. `unet-single` keeps only one conv per block: half
  the parameters, and less capacity at each scale.
* **residual** (`--residual`): the block computes `relu(F(x) + x)` (with a 1x1 conv on `x`
  when the channels change). The block only has to learn a *correction* to its input, and
  the gradient has a direct path through the `+ x`, which helps deeper networks train.
  Note: in the coursework notebook the flag called `is_resnet` actually switched the skip
  connections between encoder and decoder, not residual blocks; both exist here, under
  their own names.

### 3.2 Down-sampling: max-pool or strided convolution
""")
code("""
print(inspect.getsource(models.Down))
""")
md("""
* **max-pool 2x2**: keeps the strongest response in each 2x2 window. No weights, and a
  small shift of the input does not change the output, which is what we want for "is
  there an edge here".
* **strided conv** (`unet-strided`): a 3x3 conv with stride 2 *learns* how to summarise
  each window. More flexible, but it adds weights, and early in training it is a random
  projection that can throw away exactly the strong activations that max-pool keeps.

### 3.3 Up-sampling and the skip connection
""")
code("""
print(inspect.getsource(models.Up))
""")
md("""
Concatenation (`torch.cat`) keeps both sources intact and lets the next conv decide how
to mix them. Adding them (as in ResNet-style fusion) or taking a max would force the
encoder and decoder features into the same space and lose information.

### 3.4 Shapes through the network
""")
code("""
net = models.build_model(ModelConfig.from_preset("unet", base=BASE)).eval()
stages = [("stem", net.stem)] + [(f"down{i+1}", m) for i, m in enumerate(net.downs)] \\
       + [(f"up{len(net.ups)-i}", m) for i, m in enumerate(net.ups)] + [("head", net.head)]
hooks = [m.register_forward_hook(lambda m, i, o, n=n: print(f"{n:6s} {tuple(i[0].shape[1:])!s:18s}"
         + (f"+ skip {tuple(i[1].shape[1:])!s:18s}" if len(i) > 1 else " " * 25) + f"-> {tuple(o.shape[1:])}"))
         for n, m in stages]
with torch.no_grad():
    net(torch.randn(1, 3, SIZE, SIZE))
for h in hooks:
    h.remove()
print(f"{models.count_parameters(net) / 1e6:.2f} M parameters")
""")
md("""
### 3.5 The other end of the spectrum: a fully connected "segmenter"

The coursework also built a network out of `Linear` layers only (`fcn` here): flatten the
image, a stack of dense layers, reshape to H x W. Every output pixel is connected to every
input pixel with its own weight, so the model knows nothing about neighbourhoods, has to
learn the same edge detector separately at every position, and its first layer alone has
`3 * H * W * hidden` weights. (The coursework called it FCN; the usual meaning of FCN,
*Fully Convolutional Network*, is the opposite: no dense layers at all.)
""")
code("""
for name in PRESETS:
    m = models.build_model(ModelConfig.from_preset(name, base=BASE, img_size=SIZE))
    print(f"{name:14s} {models.count_parameters(m) / 1e6:7.2f} M parameters")
""")

md("""
## 4. Losses and metrics

* **BCE** treats each pixel as an independent yes/no; when the object is small, the many
  easy background pixels dominate the average.
* **Dice loss** `1 - 2|P∩Y| / (|P|+|Y|)` scores the overlap of the whole mask and does not
  care how much background there is. BCE + Dice (the default) combines BCE's smooth
  per-pixel gradients with Dice's focus on overlap.

Metrics come from the pixel confusion matrix: **IoU** = TP / (TP+FP+FN), **Dice** = 2TP /
(2TP+FP+FN) (= F1), precision, recall, pixel accuracy. Pixel accuracy is misleading: if
70% of the pixels are background, predicting nothing scores 70%. Models are selected on
validation **IoU**.
""")
code("""
print(inspect.getsource(losses.SegLoss))
print(inspect.getsource(metrics.Confusion.summary))
""")

md("""
## 5. Train U-Net on Pets

The loop: AdamW, warm-up then cosine learning rate, gradient clipping at 5, BCE + Dice,
validation IoU every epoch, `best.pt` kept by validation IoU. The curves are redrawn after
every epoch; the third panel is the parameter update ratio ‖ΔW‖/‖W‖ from the coursework
(around 1e-3 is the healthy range: much higher and training is unstable, much lower and
it barely moves).
""")
code("""
def live(history):
    clear_output(wait=True)
    viz.plot_curves(history, WORK / "live.png", title=f"epoch {history[-1]['epoch']}")
    display(ShowImage(str(WORK / "live.png")))
    h = history[-1]
    print(f"epoch {h['epoch']}: val IoU {h['val_iou']:.4f}  Dice {h['val_dice']:.4f}  ({h['time_s']:.0f}s)")

mcfg = ModelConfig.from_preset("unet", base=BASE)
tcfg = TrainConfig(data_dir=str(DATA), out_dir=str(WORK / "runs/unet"), img_size=SIZE, epochs=EPOCHS,
                   batch_size=BATCH, lr=LR, loss=LOSS, device=DEVICE, threads=2)
unet_summary = train(mcfg, tcfg, on_epoch=live, verbose=False)
print(json.dumps(unet_summary["test"]["at_threshold"], indent=1))
display(ShowImage(str(WORK / "runs/unet/predictions.png")))
""")

md("""
## 6. Compare every variant on the same data

Each switch of the coursework, one at a time, against the same `unet` baseline:

| model | what changes vs `unet` | question it answers |
|---|---|---|
| `unet-strided` | strided conv instead of max-pool | is learned down-sampling better? |
| `unet-single` | one conv per block instead of two | does the double conv matter? |
| `unet-residual` | residual blocks | do shortcuts inside the blocks help? |
| `segnet` | no skip connections | how much do the skips bring? |
| `fcn` | no convolutions at all | what does the conv encoder-decoder structure buy? |
""")
code("""
rows = [unet_summary]
for name in ["unet-strided", "unet-single", "unet-residual", "segnet", "fcn"]:
    print(f"=== {name} ===")
    tc = TrainConfig(data_dir=str(DATA), out_dir=str(WORK / f"runs/{name}"), img_size=SIZE, epochs=CMP_EPOCHS,
                     batch_size=BATCH, lr=LR if name != "fcn" else 1e-3, loss=LOSS, device=DEVICE, threads=2)
    rows.append(train(ModelConfig.from_preset(name, base=BASE), tc, verbose=False))
    t = rows[-1]["test"]["at_threshold"]
    print(f"{name}: test IoU {t['iou']:.4f} Dice {t['dice']:.4f} ({rows[-1]['train_time_s'] / 60:.1f} min)")

table = "| model | params | epochs | test IoU | test Dice | pixel acc | precision | recall |\\n"
table += "|---|---|---|---|---|---|---|---|\\n"
for r in rows:
    t = r["test"]["at_threshold"]
    table += (f"| `{r['model']}` | {r['params'] / 1e6:.2f} M | {r['epochs']} | {t['iou']:.3f} | {t['dice']:.3f} | "
              f"{t['pixel_acc']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} |\\n")
display(Markdown(table))
""")
code("""
fig, ax = plt.subplots(figsize=(8, 4))
for r in rows:
    h = json.loads((WORK / f"runs/{r['model']}/history.json").read_text())
    ax.plot([x["epoch"] for x in h], [x["val_iou"] for x in h], "o-", ms=3, label=r["model"])
ax.set(xlabel="epoch", ylabel="val IoU", ylim=(0, 1)); ax.grid(alpha=.3); ax.legend(); plt.show()
""")

md("""
## 7. Thresholds

The model gives probabilities; the 0.5 cut-off is a choice. A lower threshold calls more
pixels "pet" (recall up, precision down), a higher one fewer. The threshold is picked
on the **validation** split, and the test split is scored with it, so the test number is
not tuned on itself. (The coursework swept `range(40, 85, 5)`, i.e. thresholds 40 ... 80
instead of 0.40 ... 0.80, so every pixel was predicted background in that sweep.)
""")
code("""
model, _, _ = load_checkpoint(WORK / "runs/unet/best.pt", DEVICE)
val_ds = data.SegDataset.from_folder(DATA, "val", SIZE)
probs, targets = metrics.collect_probs(model, torch.utils.data.DataLoader(val_ds, batch_size=64), DEVICE)
sweep = metrics.threshold_sweep(probs, targets)
viz.plot_threshold_sweep(sweep, WORK / "thresholds.png", title="unet: threshold sweep on val")
display(ShowImage(str(WORK / "thresholds.png")))
best = max(sweep, key=lambda s: s["iou"])
print(f"best val IoU {best['iou']:.4f} at threshold {best['threshold']:.2f}; "
      f"test IoU at it: {unet_summary['test']['at_best_threshold']['iou']:.4f}")
""")

md("""
## 8. Predictions

Rows: image, true mask, predicted probability, and the error overlay (green = correct
pet pixels, red = false positives, blue = missed pet pixels).
""")
code("""
test_ds = data.SegDataset.from_folder(DATA, "test", SIZE)
for name in ["unet", "segnet", "fcn"]:
    m, _, _ = load_checkpoint(WORK / f"runs/{name}/best.pt", DEVICE)
    viz.prediction_grid(m, test_ds, WORK / f"pred_{name}.png", n=8, title=name, device=DEVICE)
    display(ShowImage(str(WORK / f"pred_{name}.png")))
""")

md("""
## 9. Conclusions

These were the coursework's conclusions (on the private competition data), in the
author's words:

> 1. **Max Pooling vs. Strided Convolution:** Max pooling is more effective for
>    down-sampling in U-Net, as it preserves critical features better than strided
>    convolution.
> 2. **Skip Connections:** Skip connections are essential for merging multi-scale
>    information. Removing them slightly reduces performance, and concatenation proves to
>    be the best fusion method.
> 3. **Auto-Encoder Structure vs. FCN:** The encoder–decoder (auto-encoder) architecture
>    significantly outperforms an FCN that maintains input resolution. Its ability to
>    capture both global context and fine details makes it superior for segmentation tasks.
> 4. **Threshold Selection for Inference:** A fixed threshold (e.g., 0.5) provides a
>    reasonable balance between precision and recall, but non-optimal thresholds can degrade
>    performance. To achieve the best trade-off, precision and recall must be carefully
>    tuned—either through threshold optimization or improved loss functions—based on
>    specific task requirements.

What to check against the table above:

* The strided-conv collapse of the coursework (recall 0) came with a learning rate of
  4e-3 and no BatchNorm/ReLU after the strided conv; here the strided conv has BN + ReLU
  and the schedule warms up, so compare whether max-pool still wins and by how much.
* The coursework's FCN ended with BatchNorm + ReLU, so its logits were never negative
  (every probability >= 0.5); here it ends with a plain Linear. It should still lose
  clearly, which is the real point (no locality, no hierarchy).
* The coursework applied 0.75 during *training-time validation* and then swept thresholds
  40 ... 80; the sweep here is on probabilities and picked on validation.
""")

md("## 10. Download")
code("""
import shutil
for name in ["unet", "unet-residual"]:
    print(WORK / f"runs/{name}/best.pt")
shutil.make_archive(str(WORK / "unetseg_runs"), "zip", WORK / "runs")
print("zip:", WORK / "unetseg_runs.zip", "- load a checkpoint with unetseg.train.load_checkpoint or unetseg-predict")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
}
nbf.validate(nb)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
