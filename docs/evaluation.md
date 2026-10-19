# Evaluation

```bash
unetseg-eval runs/unet/best.pt                    # test split, threshold 0.5 + the best val threshold
unetseg-eval runs/unet/best.pt --split val --threshold 0.4
```

`unetseg-train` already runs this on `best.pt` at the end of training. Outputs, next to
the checkpoint (or in `--out`):

| file | what |
|---|---|
| `metrics.json` | all metrics at `--threshold`, the best validation threshold and the test metrics at it, the confusion matrix, the full validation sweep |
| `thresholds.png` | IoU, Dice, precision, recall vs threshold (validation split) |
| `confusion.png` | the pixel confusion matrix, counts and percentages |
| `predictions.png` | image, true mask, probability map, TP / FP / FN overlay with per-image IoU |

## The metrics, and which to read

All of them come from one confusion matrix summed over every pixel of the split
(`metrics.Confusion`):

| metric | formula | reads as |
|---|---|---|
| **IoU** (Jaccard) | TP / (TP + FP + FN) | overlap of predicted and true object, divided by their union. **The number used to select checkpoints and compare models.** |
| **Dice** | 2TP / (2TP + FP + FN) | the same idea, more lenient (Dice = 2 IoU / (1 + IoU)); equals F1 on pixels |
| pixel accuracy | (TP + TN) / all | dominated by the background: on the shapes (12% object) an empty prediction scores 0.88 |
| precision | TP / (TP + FP) | when the model says "object", how often it is right |
| recall | TP / (TP + FN) | how much of the object it finds |
| mean IoU per image | mean over images of the image IoU | small objects count as much as large ones (in `metrics.json`) |

IoU and Dice ignore TN on purpose: the background is easy, and counting it hides
failures. The coursework reported pixel accuracy and selected models on it, which is why
its strided-conv run looked like 76% accuracy while it found no object at all.

## Thresholds

The model outputs probabilities. `unetseg-eval` collects them once and scores thresholds
0.10, 0.15, ..., 0.90 on the **validation** split, picks the one with the best IoU, and
reports the test split at 0.5 and at that threshold. Choosing the threshold on the test
split would tune on the number being reported.

On every run below the best validation threshold was between 0.45 and 0.55 and changed
test IoU by less than 0.01. With BCE + Dice the probabilities are pushed close to
0 or 1, so 0.5 is already near-optimal (see `assets/shapes_unet_thresholds.png`: the
curves are flat in the middle). The exception is the fully connected model, whose
probabilities are all mediocre: its best threshold was 0.20, and moving there raises IoU
from 0.10 to 0.17. The coursework's threshold study (0.75 worse than 0.5) points the
same way, but its sweep over `range(40, 85, 5)` could not show it.

## The runs behind the README results

Short CPU runs on a shared, loaded 8-core machine (other trainings running at the same
time), 1 thread per run except `unet` on shapes (2 threads), so the times are only
indicative.

### Synthetic shapes

```bash
unetseg-data shapes                    # 800 / 100 / 100, 96 px, object = 12% of pixels
python scripts/compare_all.py --data data/shapes --epochs 6 --max-train 600 --threads 1 --out runs/shapes
```

96 px, the first 600 training images, 100 validation / 100 test images, 6 epochs, batch 16,
BCE + Dice, AdamW lr 2e-3 with warm-up + cosine, `--base 16 --depth 4`.

| model | params | test IoU | test Dice | pixel acc | precision | recall | best val thr. | test IoU at it | val IoU per epoch | time |
|---|---|---|---|---|---|---|---|---|---|---|
| `unet-single` | 0.96 M | **0.660** | **0.796** | 0.960 | 0.834 | 0.760 | 0.45 | 0.657 | .16 .38 .52 .54 .63 .64 | 5.8 min |
| `unet-residual` | 2.03 M | 0.633 | 0.775 | 0.955 | 0.804 | 0.749 | 0.55 | 0.636 | .18 .21 .44 .59 .63 .64 | 10.7 min |
| `unet` | 1.94 M | 0.560 | 0.718 | 0.943 | 0.728 | 0.708 | 0.50 | 0.560 | .15 .25 .40 .49 .55 .60 | 7.6 min |
| `segnet` | 1.75 M | 0.489 | 0.657 | 0.931 | 0.676 | 0.639 | 0.45 | 0.482 | .03 .16 .29 .36 .45 .49 | 6.9 min |
| `unet-strided` | 2.14 M | 0.422 | 0.593 | 0.924 | 0.659 | 0.539 | 0.50 | 0.422 | .13 .16 .29 .39 .44 .46 | 10.8 min |
| `fcn` | 19.21 M | 0.102 | 0.184 | 0.837 | 0.190 | 0.179 | 0.20 | 0.167 | .12 .14 .09 .08 .08 .07 | 2.2 min |

Reading it against the coursework's four questions:

1. **Max-pool vs strided conv:** max-pool wins (0.560 vs 0.422), with fewer parameters.
   The strided model did not collapse here (BN + ReLU after the strided conv, warm-up),
   but it learns more slowly and misses whole small objects (first row of the comparison
   figure). Same conclusion as the coursework, smaller gap.
2. **Skip connections:** `unet` 0.560 vs `segnet` 0.489 (within single-run noise, see
   [Before / after figures](#before--after-figures), but visible in the pictures). Without skips the masks are
   rounded blobs and small triangles are lost or smeared (figure, column `segnet`).
   Same conclusion as the coursework.
3. **Encoder-decoder vs fully connected:** 0.56 vs 0.10, with 10x the parameters. The
   fully connected model learns roughly *where* objects tend to be and paints noise there;
   its validation IoU even falls after epoch 2 (it starts memorising the 600 training
   images). Same conclusion, much larger gap than the coursework measured.
4. **Single vs double conv, residual:** after 6 epochs the single-conv U-Net and the
   residual U-Net are *ahead* of the plain double-conv U-Net. This is a statement about
   speed of learning in a short run, not about final quality: the double-conv model is
   deeper per block and still climbing (0.49 -> 0.55 -> 0.60 over the last three epochs),
   and the residual path is exactly what makes a deeper block train faster. On a GPU
   run with 25 epochs (the notebook) the order between these three is expected to tighten.

### Oxford-IIIT Pet (subset)

```bash
unetseg-data pets --max-images 1200 --size 128      # random 1 200 photos, 960 / 120 / 120
python scripts/compare_all.py --data data/pets --size 96 --models unet unet-residual \
       --epochs 6 --max-train 600 --threads 1 --out runs/pets
```

Same settings, images resized to 96 px, the first 600 training photos, pet = 43% of the
pixels. The two models were chosen before the shapes results were in (the plain U-Net
baseline and the residual variant); `unet-single` would be the next one to try.

| model | params | test IoU | test Dice | pixel acc | precision | recall | best val thr. | test IoU at it | time |
|---|---|---|---|---|---|---|---|---|---|
| `unet-residual` | 2.03 M | **0.679** | **0.809** | 0.829 | 0.767 | 0.855 | 0.45 | 0.676 | 8.1 min |
| `unet` | 1.94 M | 0.641 | 0.781 | 0.800 | 0.727 | 0.844 | 0.50 | 0.641 | 6.6 min |
| `segnet` | 1.75 M | 0.630 | 0.773 | 0.800 | 0.744 | 0.803 | 0.45 | 0.630 | 3.8 min |
| `fcn` | 19.21 M | 0.524 | 0.688 | 0.729 | 0.671 | 0.705 | 0.35 | 0.525 | 1.2 min |

(`segnet` and `fcn` were added later with the same command and `--models segnet fcn`.)

Recall is high and precision lower for the U-Nets. With 600 photos and 6 epochs the
models find the animal but also paint background regions of similar colour (doors and
furniture in the comparison figure). Pixel accuracy (0.80) is far from IoU here for the
opposite reason to the shapes: the object is large, so false positives cost a lot of IoU.

The order is the same as on the shapes, but the gaps are different:

- **Skip connections:** worth only +0.01. A pet is one large blob, and at 96 px few
  pixels sit on its outline.
- **The fully connected model:** reaches 0.52, because the average Pets mask (a centred
  blob) is already a decent guess. `assets/unet_segnet_fcn_pets.png` shows it predicting
  that blob whatever the animal's pose.

These numbers are what a CPU allows; the notebook trains on all 7 390 photos at 160 px.

## Before / after figures

`scripts/before_after.py train` re-trains `unet` with the `on_model` hook of `train()`. The
hook records the predicted probabilities for 4 fixed test images before training
(epoch 0) and after every epoch, into `runs/progress/<name>/progress.npz`.
`scripts/before_after.py plot` turns those into:

- `assets/before_after_*.png`: epoch 0, the middle epoch and the last epoch;
- `assets/progress_*.gif`: one frame per epoch, with the mean IoU in the caption,
  about 0.3-0.5 MB each;
- `assets/unet_segnet_fcn_*.png`: the same images through the saved best checkpoints.

The shapes re-run (1 thread) ended at test IoU 0.467, and the original run in the table
(2 threads) at 0.560. Floating-point summation order differs with the thread count, and
in a 6-epoch run that is enough to move IoU by 0.09. The Pets re-run reproduced 0.641
exactly. Read differences smaller than about 0.1 between the shapes rows as noise.
