# Tests

```bash
OMP_NUM_THREADS=1 pytest -q          # 29 tests, ~30 s on one CPU core
```

| file | what it checks |
|---|---|
| `test_models.py` | every preset gives one logit per pixel and backpropagates; presets and CLI overrides set the right switches; skip connections double the decoder input channels; max-pool has no weights and strided conv has; single vs double conv count; residual shortcut is a 1×1 conv only when channels change; odd input sizes (50×50) work; the fully connected model has no convs, can output negative logits, and rejects other sizes |
| `test_losses_metrics.py` | Dice loss is ~0 for a perfect prediction; `bce+dice` is the sum; IoU, Dice, F1, precision, recall, pixel accuracy and the confusion matrix on a hand-computed example; pixel accuracy rewards an empty prediction but IoU does not; empty-vs-empty image IoU is 1; the threshold sweep finds the right threshold and rejects the coursework's `range(40, 85, 5)` |
| `test_data.py` | the shapes generator is deterministic, binary and has a reasonable object fraction; trimap border counts as pet; the augmentation keeps image and mask aligned and the mask binary; the albumentations path works (or falls back); splits are read sorted, train is augmented and val/test are not (the coursework bug) |
| `test_train.py` | **a small U-Net overfits one batch of 8 shape images to IoU > 0.8**; `train()` writes every file of a run folder, the checkpoint round-trips and `predict_image` returns a probability map at the input resolution; the fully connected model trains through the same loop; CLI switches override the preset; the `on_model` hook sees the untrained model (epoch 0) and every epoch |

The overfit test is the one to keep if everything else is deleted: if a model cannot
memorise eight images, something between the data, the loss and the optimiser is
broken, whatever the other numbers say.

The examples (`examples/*.py`) and the notebook are run by hand, see their READMEs.
