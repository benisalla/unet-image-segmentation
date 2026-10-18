# legacy/

`Image_Segmentation.ipynb` is the original notebook from the École Centrale computer-vision
coursework (binary segmentation on a private competition dataset, trained on Colab), kept
verbatim for reference. The maintained code is the package in [`src/unetseg`](../src/unetseg).

## What the notebook did

* `SegDataset` over `images/` + `masks/`, albumentations augmentations, 80 / 10 / 10 split.
* One `UNet` class with three switches: `is_resnet` (encoder-decoder skip connections),
  `is_max_pool` (max-pool or stride-2 conv), `is_double` (one or two convs per block),
  64 -> 1024 channels, 31 M parameters at its largest. With `is_resnet=False` it is a
  SegNet-like encoder-decoder.
* An `FCN` made of `Linear` layers only (flatten, dense stack, reshape).
* BCE loss, AdamW (lr 4e-3), exponential lr decay, gradient clipping, a live three-panel
  plot (loss, pixel accuracy, update ratio), early stopping on validation pixel accuracy.
* Six runs: U-Net with skips / without skips / strided conv / single conv / threshold
  0.75 / FCN, each followed by a test report (accuracy, precision, recall, F1, confusion
  matrix) and a written discussion. Those discussions are kept in
  [docs/architecture.md](../docs/architecture.md#what-the-coursework-concluded) and in the
  notebook's conclusion section.

## Problems found, and how the package handles them

| Notebook | Effect | Now |
|---|---|---|
| `test_set.dataset.img_transform = val_transform` | the dataset stores its transform in `self.augmentation`, so this line set an unused attribute; and the three `random_split` subsets share **one** dataset object, so it could not have worked anyway: validation and test were evaluated on **augmented** (flipped, rotated, shifted) images | each split is its own `SegDataset`; val/test have `augment=None` (tested) |
| `is_resnet` flag and the "Residual Connections" headings | the flag switched the U-Net skip connections (concatenation); there were no residual blocks anywhere | `--skip / --no-skip` for skip connections, `--residual` for real residual blocks (`relu(F(x) + Wx)`) |
| `FCN` ended with `D2FC` = Linear -> BatchNorm -> **ReLU** | logits >= 0, so every probability was >= 0.5: "background" could only be said by a ReLU output of exactly 0 (`sigmoid(0) > 0.5` is false), the BCE of background pixels could not go below ln 2, and any threshold under 0.5 made the whole image object | the last layer is a plain `Linear` (a test checks negative logits are possible) |
| threshold sweep `for threshold in range(40, 85, 5)` | thresholds 40, 45, ..., 80 instead of 0.40 ... 0.80: `sigmoid > 40` is never true, every pixel is background | `threshold_sweep` works on probabilities in (0, 1), rejects values outside, and picks the threshold on the **validation** split |
| "Variant 5" heading says *One Single Convolutional Layer* | the code used `is_double=True` | presets are named after what they build |
| `best_val_acc` and `patience_counter` set once, before all six runs; same `best_model_path` | after the first run, later runs only saved a checkpoint if they beat the *previous model's* accuracy; and the model tested was always the **last** epoch, the saved best was never reloaded | each run has its own folder, `best.pt` is chosen by that run's val IoU and is the one evaluated |
| early stopping and model selection on **pixel accuracy** | with ~75% background, predicting nothing already scores ~75% (the strided run: accuracy 75.99% with recall 0) | selection on validation IoU; pixel accuracy only reported |
| `if step >= n_step: break` with `n_step = 10` and `shuffle=True` on the validation loader | each "epoch" was 11 batches of training and 11 random validation batches: noisy, non-repeatable validation | full passes over train and val; val/test loaders are not shuffled |
| strided `DownConv`: `Conv2d(stride=2)` with no BatchNorm / ReLU, then the block | a purely linear down-sampling step before each block | `conv -> BN -> ReLU` with stride 2, then the block |
| update ratio = `‖lr · grad‖ / ‖W‖`, computed before clipping | that is not the step AdamW takes | measured as `‖W_after − W_before‖ / ‖W_before‖` after the optimizer step |
| `os.listdir` order for the image list | split depends on the file system order, not reproducible across machines | file names sorted before any split |
| `import torch` twice, `torchmetrics` only for four metrics, `im_size` unused in `SegDataset` | clutter | metrics computed from one confusion matrix in `metrics.py`, no extra dependency |
| no tests | nothing caught the problems above | 28 pytest checks, including a "U-Net overfits one batch" test |
