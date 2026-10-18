# Training

```bash
unetseg-train --model unet --data data/shapes --epochs 8          # ~5 min on a free CPU core
unetseg-train --model segnet --data data/pets --size 128 --epochs 6
unetseg-train --model unet --no-skip --down strided --residual    # any combination of switches
```

Every run writes one folder (`runs/<model>` or `--out`):

| file | what |
|---|---|
| `best.pt` | the checkpoint with the highest **validation IoU** (weights + model and train config) |
| `last.pt` | the last epoch |
| `history.json` | one row per epoch: train loss, val loss / IoU / Dice / pixel acc / precision / recall, update ratio, lr, seconds |
| `curves.png` | loss, validation metrics and update ratio, redrawn after every epoch (open it while training) |
| `summary.json` | parameter count, time, best val IoU, configs, test metrics |
| `metrics.json`, `thresholds.png`, `confusion.png`, `predictions.png` | the test evaluation of `best.pt`, see [evaluation.md](evaluation.md) |

## The loop, and why each piece is there

```
for epoch:
    for (x, y) in train loader (shuffled, augmented):
        z = model(x)                       # B×1×H×W logits
        loss = BCE(z, y) + Dice(z, y)      # --loss bce | dice | bce+dice
        loss.backward()
        clip_grad_norm_(params, 5)         # one bad batch cannot blow the weights up
        AdamW.step(); lr_schedule.step()
    validate at --threshold -> IoU, Dice, ...
    if val IoU improved: save best.pt
```

* **AdamW, lr 2e-3, weight decay 1e-4.** Adam adapts the step per parameter, which
  matters with BatchNorm and layers of very different sizes; AdamW applies the weight
  decay directly to the weights instead of mixing it into the gradient. The coursework
  used 4e-3, which made the strided-conv run collapse.
* **Warm-up then cosine.** The learning rate rises linearly over the first 50 steps
  (or 10% of the run) and then follows a half cosine down to 2% of its peak. Warm-up
  avoids large random steps while BatchNorm statistics are still meaningless; the cosine
  ends the run with small, fine steps. (The coursework decayed `lr · 0.98^epoch`, which
  over a few epochs barely changes anything.)
* **Model selection on validation IoU,** not pixel accuracy: with 88% background
  (shapes), a model that outputs nothing has 88% accuracy and 0 IoU.
* **Update ratio** `‖W_after − W_before‖ / ‖W_before‖`, averaged over the parameter
  tensors every 10 steps (third panel of `curves.png`). Around 1e-3 per step is the
  usual healthy value; 1e-1 means the weights are being rewritten each step (lr too
  high), 1e-5 means nothing moves (lr too low or dead units). It is a cheap way to see
  whether the learning rate is sane without a sweep.
* **Augmentation** only on the training split (flips, random 90° rotation, small affine;
  see [data.md](data.md)). `--augment none` turns it off, useful to see overfitting.

## Sizes and speed

| setting | default | note |
|---|---|---|
| `--size` | 96 | must be divisible by `2**depth` for exact shapes (others are padded) |
| `--base` | 16 | channels of the first block; the coursework used 64 (×16 parameters) |
| `--depth` | 4 | down-sampling steps; bottleneck is `size / 2**depth` |
| `--batch-size` | 16 | |
| `--threads` | 1 | sets `torch.set_num_threads` and `OMP_NUM_THREADS` |
| `--max-train` | all | use only the first N training images (quick experiments) |

On one free CPU core a 96 px U-Net step (batch 16, forward + backward) takes ~0.7 s;
an epoch of 800 shapes ~35 s. `unet-single` is twice as fast, `fcn` is almost free at
96 px (but has 10× the parameters). On a GPU use the notebook (`--base 32`, 160 px,
the full Pets dataset).

## Reproducing the README results

```bash
unetseg-data shapes                                   # 800 / 100 / 100, 96 px
python scripts/compare_all.py --data data/shapes --epochs 6 --max-train 600 --threads 2 --out runs/shapes
unetseg-data pets --max-images 1200 --size 128        # ~3 min, 41 MB on disk
python scripts/compare_all.py --data data/pets --size 128 --models unet unet-residual \
       --epochs 6 --threads 2 --out runs/pets
```
