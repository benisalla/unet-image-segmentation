# Examples

Six short scripts that use the `unetseg` package from Python, in the order you would use
it. Each one is a single command, runs on a CPU with small defaults, and writes its
figures to `examples/output/`. Run them from the repository root after
`pip install -e ".[dev]"`.

| script | what it shows | time (1 CPU core) | needs |
|---|---|---|---|
| [01_make_data_and_look.py](01_make_data_and_look.py) | writes 400/60/60 synthetic shapes (96 px) and draws samples and four augmentations of one image/mask pair | ~3 s | nothing |
| [02_train_unet.py](02_train_unet.py) | `ModelConfig` + `TrainConfig` + `train()`: the default U-Net for 4 epochs, with curves, checkpoint and test metrics | ~2-4 min | 01 |
| [03_evaluate_and_thresholds.py](03_evaluate_and_thresholds.py) | collects probabilities once, sweeps the threshold 0.1 ... 0.9 (IoU, Dice, precision, recall) and draws the confusion matrix | ~5 s | 02 |
| [04_predict_and_overlay.py](04_predict_and_overlay.py) | `predict_image` on PNG files (as `unetseg-predict` does) with prediction and TP/FP/FN overlays | ~5 s | 02 |
| [05_switches_side_by_side.py](05_switches_side_by_side.py) | every preset: switches, parameter count, forward time; then the tensor shape at every stage of the U-Net | ~5 s | nothing |
| [06_skip_vs_no_skip.py](06_skip_vs_no_skip.py) | one coursework question on a CPU: `unet` vs `segnet` (no skip connections), 3 epochs each, side by side | ~4-6 min | 01 |

```bash
python examples/01_make_data_and_look.py
python examples/02_train_unet.py
python examples/03_evaluate_and_thresholds.py
python examples/04_predict_and_overlay.py
python examples/05_switches_side_by_side.py
python examples/06_skip_vs_no_skip.py
```

The same steps as CLI commands are in the README's quick start (`unetseg-data`,
`unetseg-train`, `unetseg-eval`, `unetseg-predict`); the Kaggle/Colab notebook does them
on the full Pets dataset with a GPU.
